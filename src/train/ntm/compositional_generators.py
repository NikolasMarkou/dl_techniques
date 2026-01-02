"""
Compositional Generalization Benchmark Generators.

Generators for SCAN-style and COGS-style compositional generalization
tasks that test systematic recombination of learned primitives.
"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import ScanTaskConfig


class ScanSplit(Enum):
    """SCAN dataset split types."""
    SIMPLE = "simple"
    LENGTH = "length"
    ADD_PRIM_JUMP = "add_prim_jump"
    ADD_PRIM_TURN_LEFT = "add_prim_turn_left"
    TEMPLATE_AROUND_RIGHT = "template_around_right"


@dataclass
class ScanSample:
    """A single SCAN command-action pair.
    
    :param command: Input command string.
    :param actions: Output action sequence.
    :param command_tokens: Tokenized command.
    :param action_tokens: Tokenized actions.
    """
    command: str
    actions: str
    command_tokens: List[str]
    action_tokens: List[str]


class ScanGenerator:
    """Generator for SCAN compositional generalization benchmark.
    
    SCAN tests systematic recombination of navigation commands,
    evaluating compositional generalization capabilities.
    
    :param config: Configuration for SCAN generation.
    
    Example::
    
        config = ScanTaskConfig(split_type='length')
        generator = ScanGenerator(config)
        train_data, test_data = generator.generate_split()
    """
    
    # Primitive commands
    PRIMITIVES = ["walk", "run", "jump", "look"]
    DIRECTIONS = ["left", "right"]
    MODIFIERS = ["twice", "thrice"]
    CONJUNCTIONS = ["and", "after"]
    AROUND = "around"
    OPPOSITE = "opposite"
    
    # Action mappings
    PRIMITIVE_ACTIONS = {
        "walk": ["WALK"],
        "run": ["RUN"],
        "jump": ["JUMP"],
        "look": ["LOOK"]
    }
    
    TURN_ACTIONS = {
        "left": "LTURN",
        "right": "RTURN"
    }
    
    def __init__(self, config: ScanTaskConfig) -> None:
        """Initialize the SCAN generator.
        
        :param config: SCAN task configuration.
        """
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
        
        # Build vocabulary
        self._build_vocabulary()
    
    def _build_vocabulary(self) -> None:
        """Build command and action vocabularies."""
        self.command_vocab = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2
        }
        idx = 3
        for word in self.PRIMITIVES + self.DIRECTIONS + self.MODIFIERS + \
                    self.CONJUNCTIONS + [self.AROUND, self.OPPOSITE, "turn"]:
            if word not in self.command_vocab:
                self.command_vocab[word] = idx
                idx += 1
        
        self.action_vocab = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "WALK": 3,
            "RUN": 4,
            "JUMP": 5,
            "LOOK": 6,
            "LTURN": 7,
            "RTURN": 8
        }
        
        self.inv_command_vocab = {v: k for k, v in self.command_vocab.items()}
        self.inv_action_vocab = {v: k for k, v in self.action_vocab.items()}
    
    def generate_all_samples(self) -> List[ScanSample]:
        """Generate all valid SCAN samples.
        
        :return: List of all valid command-action pairs.
        """
        samples = []
        
        # Single primitives
        for prim in self.PRIMITIVES:
            samples.append(self._create_sample(prim))
        
        # Turn commands
        for direction in self.DIRECTIONS:
            samples.append(self._create_sample(f"turn {direction}"))
        
        # Primitives with direction
        for prim in self.PRIMITIVES:
            for direction in self.DIRECTIONS:
                samples.append(self._create_sample(f"{prim} {direction}"))
        
        # With modifiers (twice, thrice)
        for prim in self.PRIMITIVES:
            for mod in self.MODIFIERS:
                samples.append(self._create_sample(f"{prim} {mod}"))
        
        # Direction + modifier
        for prim in self.PRIMITIVES:
            for direction in self.DIRECTIONS:
                for mod in self.MODIFIERS:
                    samples.append(self._create_sample(f"{prim} {direction} {mod}"))
        
        # Around commands
        for prim in self.PRIMITIVES:
            for direction in self.DIRECTIONS:
                samples.append(self._create_sample(f"{prim} around {direction}"))
        
        # Opposite commands
        for prim in self.PRIMITIVES:
            for direction in self.DIRECTIONS:
                samples.append(self._create_sample(f"{prim} opposite {direction}"))
        
        # Compound commands with conjunctions
        simple_commands = [p for p in self.PRIMITIVES] + \
                         [f"{p} {d}" for p in self.PRIMITIVES for d in self.DIRECTIONS]
        
        for cmd1 in simple_commands[:10]:  # Limit for tractability
            for cmd2 in simple_commands[:10]:
                for conj in self.CONJUNCTIONS:
                    samples.append(self._create_sample(f"{cmd1} {conj} {cmd2}"))
        
        return samples
    
    def _create_sample(self, command: str) -> ScanSample:
        """Create a SCAN sample from a command string.
        
        :param command: Input command.
        :return: ScanSample with command and corresponding actions.
        """
        actions = self._execute_command(command)
        return ScanSample(
            command=command,
            actions=" ".join(actions),
            command_tokens=command.split(),
            action_tokens=actions
        )
    
    def _execute_command(self, command: str) -> List[str]:
        """Execute a SCAN command to produce action sequence.
        
        :param command: Input command string.
        :return: List of action tokens.
        """
        tokens = command.split()
        
        # Handle conjunctions
        if "and" in tokens:
            idx = tokens.index("and")
            left = self._execute_command(" ".join(tokens[:idx]))
            right = self._execute_command(" ".join(tokens[idx + 1:]))
            return left + right
        
        if "after" in tokens:
            idx = tokens.index("after")
            left = self._execute_command(" ".join(tokens[:idx]))
            right = self._execute_command(" ".join(tokens[idx + 1:]))
            return right + left  # "after" reverses order
        
        # Parse modifiers
        repeat = 1
        if "twice" in tokens:
            repeat = 2
            tokens = [t for t in tokens if t != "twice"]
        if "thrice" in tokens:
            repeat = 3
            tokens = [t for t in tokens if t != "thrice"]
        
        # Handle around
        if "around" in tokens:
            idx = tokens.index("around")
            direction = tokens[idx + 1]
            base_action = tokens[0]
            turn = self.TURN_ACTIONS[direction]
            base = self.PRIMITIVE_ACTIONS[base_action][0]
            # around = turn, action, turn, action, turn, action, turn, action
            unit = [turn, base]
            return (unit * 4) * repeat
        
        # Handle opposite
        if "opposite" in tokens:
            idx = tokens.index("opposite")
            direction = tokens[idx + 1]
            base_action = tokens[0]
            turn = self.TURN_ACTIONS[direction]
            base = self.PRIMITIVE_ACTIONS[base_action][0]
            # opposite = turn, turn, action
            return ([turn, turn, base]) * repeat
        
        # Handle turn
        if tokens[0] == "turn":
            direction = tokens[1]
            return [self.TURN_ACTIONS[direction]] * repeat
        
        # Handle primitive with direction
        actions = []
        base_action = tokens[0]
        if len(tokens) > 1 and tokens[1] in self.DIRECTIONS:
            direction = tokens[1]
            actions.append(self.TURN_ACTIONS[direction])
        
        actions.extend(self.PRIMITIVE_ACTIONS[base_action])
        return actions * repeat
    
    def generate_split(
        self,
        split_type: Optional[str] = None
    ) -> Tuple[List[ScanSample], List[ScanSample]]:
        """Generate train/test split based on split type.
        
        :param split_type: Type of split. Uses config default if None.
        :return: Tuple of (train_samples, test_samples).
        """
        split = split_type or self.config.split_type
        all_samples = self.generate_all_samples()
        
        if split == "simple":
            return self._simple_split(all_samples)
        elif split == "length":
            return self._length_split(all_samples)
        elif split == "add_prim_jump":
            return self._add_primitive_split(all_samples, "jump")
        elif split == "add_prim_turn_left":
            return self._add_primitive_split(all_samples, "turn left")
        else:
            return self._simple_split(all_samples)
    
    def _simple_split(
        self,
        samples: List[ScanSample],
        test_ratio: float = 0.2
    ) -> Tuple[List[ScanSample], List[ScanSample]]:
        """Random train/test split.
        
        :param samples: All samples.
        :param test_ratio: Fraction for test set.
        :return: Train and test splits.
        """
        indices = self._rng.permutation(len(samples))
        split_idx = int(len(samples) * (1 - test_ratio))
        
        train = [samples[i] for i in indices[:split_idx]]
        test = [samples[i] for i in indices[split_idx:]]
        
        return train, test
    
    def _length_split(
        self,
        samples: List[ScanSample],
        length_threshold: int = 24
    ) -> Tuple[List[ScanSample], List[ScanSample]]:
        """Split by output action sequence length.
        
        :param samples: All samples.
        :param length_threshold: Max action length for training.
        :return: Train and test splits.
        """
        train = [s for s in samples if len(s.action_tokens) <= length_threshold]
        test = [s for s in samples if len(s.action_tokens) > length_threshold]
        
        return train, test
    
    def _add_primitive_split(
        self,
        samples: List[ScanSample],
        primitive: str
    ) -> Tuple[List[ScanSample], List[ScanSample]]:
        """Split where primitive only appears alone in training.
        
        :param samples: All samples.
        :param primitive: Primitive to hold out in compositions.
        :return: Train and test splits.
        """
        train = []
        test = []
        
        for sample in samples:
            tokens = sample.command_tokens
            has_primitive = primitive in sample.command or \
                           (primitive.split()[0] in tokens if " " in primitive else primitive in tokens)
            
            # If command is just the primitive, goes to train
            if sample.command == primitive or \
               (len(tokens) <= 2 and has_primitive):
                train.append(sample)
            elif has_primitive:
                # Composed commands with primitive go to test
                test.append(sample)
            else:
                train.append(sample)
        
        return train, test
    
    def encode_samples(
        self,
        samples: List[ScanSample]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode samples as numpy arrays.
        
        :param samples: List of SCAN samples.
        :return: Tuple of (inputs, targets, input_lengths, target_lengths).
        """
        max_input_len = max(len(s.command_tokens) for s in samples) + 2  # +START, +END
        max_output_len = max(len(s.action_tokens) for s in samples) + 2
        
        inputs = np.zeros((len(samples), max_input_len), dtype=np.int32)
        targets = np.zeros((len(samples), max_output_len), dtype=np.int32)
        input_lengths = np.zeros(len(samples), dtype=np.int32)
        target_lengths = np.zeros(len(samples), dtype=np.int32)
        
        for i, sample in enumerate(samples):
            # Encode input
            inputs[i, 0] = self.command_vocab["<START>"]
            for j, token in enumerate(sample.command_tokens):
                inputs[i, j + 1] = self.command_vocab.get(token, 0)
            inputs[i, len(sample.command_tokens) + 1] = self.command_vocab["<END>"]
            input_lengths[i] = len(sample.command_tokens) + 2
            
            # Encode target
            targets[i, 0] = self.action_vocab["<START>"]
            for j, token in enumerate(sample.action_tokens):
                targets[i, j + 1] = self.action_vocab.get(token, 0)
            targets[i, len(sample.action_tokens) + 1] = self.action_vocab["<END>"]
            target_lengths[i] = len(sample.action_tokens) + 2
        
        return inputs, targets, input_lengths, target_lengths


@dataclass
class CogsExample:
    """A COGS semantic parsing example.
    
    :param sentence: Natural language input.
    :param logical_form: Target logical form.
    :param example_type: Type of example (train/gen/test).
    :param generalization_type: Type of generalization tested.
    """
    sentence: str
    logical_form: str
    example_type: str
    generalization_type: Optional[str] = None


class CogsGenerator:
    """Generator for COGS-style compositional semantic parsing.
    
    COGS tests compositional generalization through semantic parsing
    with controlled lexical and structural splits.
    
    :param vocab_size: Size of word vocabulary.
    :param random_seed: Seed for reproducibility.
    
    Example::
    
        generator = CogsGenerator()
        train, gen_test = generator.generate_split()
    """
    
    # Simplified COGS-style templates
    SUBJECTS = ["Emma", "Liam", "a dog", "the cat", "a bird", "the chef"]
    VERBS_INTRANS = ["slept", "ran", "jumped", "laughed"]
    VERBS_TRANS = ["saw", "helped", "liked", "chased"]
    VERBS_DITRANS = ["gave", "sent", "showed", "offered"]
    OBJECTS = ["a cake", "the ball", "a book", "the toy", "a flower"]
    PREPS = ["on", "beside", "near"]
    LOCATIONS = ["the table", "a bench", "the floor", "a hill"]
    
    def __init__(
        self,
        vocab_size: int = 200,
        random_seed: int = 42
    ) -> None:
        """Initialize the COGS generator.
        
        :param vocab_size: Maximum vocabulary size.
        :param random_seed: Random seed for reproducibility.
        """
        self.vocab_size = vocab_size
        self._rng = np.random.default_rng(random_seed)
        self._build_vocabulary()
    
    def _build_vocabulary(self) -> None:
        """Build word and logical form vocabularies."""
        all_words = set()
        for lst in [self.SUBJECTS, self.VERBS_INTRANS, self.VERBS_TRANS,
                    self.VERBS_DITRANS, self.OBJECTS, self.PREPS, self.LOCATIONS]:
            for item in lst:
                all_words.update(item.split())
        
        self.word_vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        for i, word in enumerate(sorted(all_words)):
            self.word_vocab[word] = i + 4
        
        # Logical form tokens
        lf_tokens = ["(", ")", ".", "*", "AND", "x", "agent", "theme", 
                     "recipient", "location", "LAMBDA"]
        for verb in self.VERBS_INTRANS + self.VERBS_TRANS + self.VERBS_DITRANS:
            lf_tokens.append(verb.upper())
        for subj in self.SUBJECTS + self.OBJECTS + self.LOCATIONS:
            for word in subj.split():
                if word not in ["a", "the"]:
                    lf_tokens.append(word.lower())
        
        self.lf_vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2}
        for i, token in enumerate(sorted(set(lf_tokens))):
            if token not in self.lf_vocab:
                self.lf_vocab[token] = len(self.lf_vocab)
    
    def generate_intransitive(self) -> List[CogsExample]:
        """Generate intransitive sentence examples.
        
        :return: List of intransitive examples.
        """
        examples = []
        for subj in self.SUBJECTS:
            for verb in self.VERBS_INTRANS:
                sentence = f"{subj} {verb}"
                # Simplified logical form
                subj_term = subj.split()[-1].lower()
                lf = f"* {subj_term} x ; {verb.upper()} . agent ( x , {subj_term} )"
                examples.append(CogsExample(
                    sentence=sentence,
                    logical_form=lf,
                    example_type="train"
                ))
        return examples
    
    def generate_transitive(self) -> List[CogsExample]:
        """Generate transitive sentence examples.
        
        :return: List of transitive examples.
        """
        examples = []
        for subj in self.SUBJECTS:
            for verb in self.VERBS_TRANS:
                for obj in self.OBJECTS:
                    sentence = f"{subj} {verb} {obj}"
                    subj_term = subj.split()[-1].lower()
                    obj_term = obj.split()[-1].lower()
                    lf = f"* {subj_term} x ; * {obj_term} y ; {verb.upper()} . agent ( x , {subj_term} ) AND theme ( x , {obj_term} )"
                    examples.append(CogsExample(
                        sentence=sentence,
                        logical_form=lf,
                        example_type="train"
                    ))
        return examples
    
    def generate_pp_modification(self) -> List[CogsExample]:
        """Generate prepositional phrase modification examples.
        
        :return: List of PP-modified examples.
        """
        examples = []
        for subj in self.SUBJECTS[:3]:
            for verb in self.VERBS_INTRANS[:2]:
                for prep in self.PREPS:
                    for loc in self.LOCATIONS[:2]:
                        sentence = f"{subj} {verb} {prep} {loc}"
                        subj_term = subj.split()[-1].lower()
                        loc_term = loc.split()[-1].lower()
                        lf = f"* {subj_term} x ; * {loc_term} y ; {verb.upper()} . agent ( x , {subj_term} ) AND location ( x , {loc_term} )"
                        examples.append(CogsExample(
                            sentence=sentence,
                            logical_form=lf,
                            example_type="train"
                        ))
        return examples
    
    def generate_split(
        self,
        lexical_holdout: Optional[str] = None,
        structural_holdout: bool = False
    ) -> Tuple[List[CogsExample], List[CogsExample]]:
        """Generate train/test split with compositional generalization tests.
        
        :param lexical_holdout: Word to hold out from training compositions.
        :param structural_holdout: Whether to hold out PP structures.
        :return: Tuple of (train_examples, test_examples).
        """
        intrans = self.generate_intransitive()
        trans = self.generate_transitive()
        pp = self.generate_pp_modification()
        
        train = []
        test = []
        
        all_examples = intrans + trans + pp
        
        if lexical_holdout:
            # Lexical generalization: hold out compositions with specific word
            for ex in all_examples:
                if lexical_holdout in ex.sentence.lower():
                    if ex.sentence.lower().strip() == lexical_holdout or \
                       len(ex.sentence.split()) <= 2:
                        train.append(ex)  # Primitive use in training
                    else:
                        ex.generalization_type = "lexical"
                        test.append(ex)
                else:
                    train.append(ex)
        elif structural_holdout:
            # Structural generalization: hold out PP modifications
            train = intrans + trans
            for ex in pp:
                ex.generalization_type = "structural"
            test = pp
        else:
            # Simple random split
            indices = self._rng.permutation(len(all_examples))
            split_idx = int(len(all_examples) * 0.8)
            train = [all_examples[i] for i in indices[:split_idx]]
            test = [all_examples[i] for i in indices[split_idx:]]
        
        return train, test
    
    def encode_examples(
        self,
        examples: List[CogsExample]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode examples as numpy arrays.
        
        :param examples: List of COGS examples.
        :return: Tuple of (encoded_sentences, encoded_logical_forms).
        """
        max_sent_len = max(len(ex.sentence.split()) for ex in examples) + 2
        max_lf_len = max(len(ex.logical_form.split()) for ex in examples) + 2
        
        sentences = np.zeros((len(examples), max_sent_len), dtype=np.int32)
        logical_forms = np.zeros((len(examples), max_lf_len), dtype=np.int32)
        
        for i, ex in enumerate(examples):
            # Encode sentence
            sentences[i, 0] = self.word_vocab["<START>"]
            for j, word in enumerate(ex.sentence.split()):
                sentences[i, j + 1] = self.word_vocab.get(word.lower(), self.word_vocab["<UNK>"])
            sentences[i, len(ex.sentence.split()) + 1] = self.word_vocab["<END>"]
            
            # Encode logical form
            logical_forms[i, 0] = self.lf_vocab["<START>"]
            for j, token in enumerate(ex.logical_form.split()):
                logical_forms[i, j + 1] = self.lf_vocab.get(token, 0)
            logical_forms[i, len(ex.logical_form.split()) + 1] = self.lf_vocab["<END>"]
        
        return sentences, logical_forms


class CFQGenerator:
    """Generator for CFQ-style compositional Freebase queries.
    
    CFQ tests compositional generalization through question-to-SPARQL
    translation with controlled compound divergence.
    
    :param random_seed: Seed for reproducibility.
    """
    
    # Simplified entity and relation templates
    ENTITIES = ["M0", "M1", "M2", "M3", "M4", "M5"]
    RELATIONS = [
        ("directed", "director"),
        ("produced", "producer"),
        ("wrote", "writer"),
        ("starred_in", "actor"),
        ("edited", "editor")
    ]
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize the CFQ generator.
        
        :param random_seed: Random seed for reproducibility.
        """
        self._rng = np.random.default_rng(random_seed)
    
    def generate_simple_queries(self, num_samples: int) -> List[Tuple[str, str]]:
        """Generate simple single-relation queries.
        
        :param num_samples: Number of queries to generate.
        :return: List of (question, sparql) tuples.
        """
        queries = []
        
        for _ in range(num_samples):
            entity = self._rng.choice(self.ENTITIES)
            rel_verb, rel_noun = self._rng.choice(self.RELATIONS)
            
            question = f"Who {rel_verb} {entity} ?"
            sparql = f"SELECT ?x WHERE {{ {entity} ns:{rel_noun} ?x }}"
            
            queries.append((question, sparql))
        
        return queries
    
    def generate_compound_queries(self, num_samples: int) -> List[Tuple[str, str]]:
        """Generate compound multi-relation queries.
        
        :param num_samples: Number of queries to generate.
        :return: List of (question, sparql) tuples.
        """
        queries = []
        
        for _ in range(num_samples):
            entity = self._rng.choice(self.ENTITIES)
            rel1_verb, rel1_noun = self._rng.choice(self.RELATIONS)
            rel2_verb, rel2_noun = self._rng.choice(self.RELATIONS)
            
            while rel2_verb == rel1_verb:
                rel2_verb, rel2_noun = self._rng.choice(self.RELATIONS)
            
            question = f"Who {rel1_verb} and {rel2_verb} {entity} ?"
            sparql = f"SELECT ?x WHERE {{ {entity} ns:{rel1_noun} ?x . {entity} ns:{rel2_noun} ?x }}"
            
            queries.append((question, sparql))
        
        return queries
    
    def generate_split(
        self,
        train_size: int = 1000,
        test_size: int = 200,
        compound_divergence: float = 0.5
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Generate train/test split with controlled compound divergence.
        
        :param train_size: Number of training samples.
        :param test_size: Number of test samples.
        :param compound_divergence: Controls compositional difficulty.
        :return: Tuple of (train_queries, test_queries).
        """
        # Training: mix of simple and some compounds
        train_simple = self.generate_simple_queries(int(train_size * 0.7))
        train_compound = self.generate_compound_queries(int(train_size * 0.3))
        train = train_simple + train_compound
        self._rng.shuffle(train)
        
        # Test: novel compound combinations
        test = self.generate_compound_queries(test_size)
        
        return train[:train_size], test[:test_size]
