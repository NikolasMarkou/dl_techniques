"""
bAbI-style Question Answering Task Generator.

Generates synthetic QA tasks testing various reasoning capabilities
including fact retrieval, counting, spatial reasoning, and path-finding.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import BabiTaskConfig


@dataclass
class BabiSample:
    """A single bAbI-style QA sample.
    
    :param story: List of story sentences.
    :param question: The question to answer.
    :param answer: The correct answer.
    :param supporting_facts: Indices of sentences supporting the answer.
    :param task_id: ID of the bAbI task (1-20).
    """
    story: List[str]
    question: str
    answer: str
    supporting_facts: List[int]
    task_id: int


class BabiGenerator:
    """Generator for bAbI-style question answering tasks.
    
    Implements a subset of the 20 bAbI tasks testing different
    reasoning capabilities for memory-augmented networks.
    
    :param config: Configuration for bAbI generation.
    
    Example::
    
        config = BabiTaskConfig(task_ids=[1, 2, 3])
        generator = BabiGenerator(config)
        samples = generator.generate(task_id=1, num_samples=100)
    """
    
    # Entity names
    NAMES = ["John", "Mary", "Sandra", "Daniel", "Bill", "Fred", "Julie", "Emma"]
    OBJECTS = ["apple", "football", "milk", "ball", "box", "key", "wallet", "phone"]
    LOCATIONS = ["garden", "kitchen", "bedroom", "bathroom", "office", "hallway"]
    DIRECTIONS = ["north", "south", "east", "west"]
    COLORS = ["red", "blue", "green", "yellow", "white", "black"]
    SIZES = ["small", "large", "tiny", "big"]
    ANIMALS = ["cat", "dog", "mouse", "lion", "wolf", "sheep"]
    
    TASK_DESCRIPTIONS = {
        1: "Single Supporting Fact",
        2: "Two Supporting Facts",
        3: "Three Supporting Facts",
        4: "Two Argument Relations",
        5: "Three Argument Relations",
        6: "Yes/No Questions",
        7: "Counting",
        8: "Lists/Sets",
        9: "Simple Negation",
        10: "Indefinite Knowledge",
        11: "Basic Coreference",
        12: "Conjunction",
        13: "Compound Coreference",
        14: "Time Reasoning",
        15: "Basic Deduction",
        16: "Basic Induction",
        17: "Positional Reasoning",
        18: "Size Reasoning",
        19: "Path Finding",
        20: "Agent's Motivations"
    }
    
    def __init__(self, config: BabiTaskConfig) -> None:
        """Initialize the bAbI generator.
        
        :param config: bAbI task configuration.
        """
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
        self._build_vocabulary()
    
    def _build_vocabulary(self) -> None:
        """Build vocabulary from all possible words."""
        all_words = set()
        all_words.update(self.NAMES)
        all_words.update(self.OBJECTS)
        all_words.update(self.LOCATIONS)
        all_words.update(self.DIRECTIONS)
        all_words.update(self.COLORS)
        all_words.update(self.SIZES)
        all_words.update(self.ANIMALS)
        
        # Action words
        actions = ["went", "to", "the", "is", "in", "picked", "up", "dropped",
                   "put", "down", "got", "grabbed", "moved", "travelled", "journeyed",
                   "yes", "no", "maybe", "where", "what", "who", "how", "many",
                   "before", "after", "and", "or", "not", "gave", "received",
                   "passed", "handed", "left", "right", "above", "below", "of",
                   "a", "an", "there", "does", "has", "have", "nothing", "none",
                   "was", "were", "are", "afraid", "scared", "hungry", "thirsty",
                   "bigger", "smaller", "than", "fit", "inside"]
        all_words.update(actions)
        
        # Numbers
        for i in range(20):
            all_words.add(str(i))
        
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        for i, word in enumerate(sorted(all_words)):
            self.word2idx[word.lower()] = i + 2
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
    
    def generate(
        self,
        task_id: int,
        num_samples: Optional[int] = None
    ) -> List[BabiSample]:
        """Generate samples for a specific bAbI task.
        
        :param task_id: Task ID (1-20).
        :param num_samples: Number of samples to generate.
        :return: List of BabiSample objects.
        :raises ValueError: If task_id is not supported.
        """
        num_samples = num_samples or self.config.num_samples_per_task
        
        generator_map = {
            1: self._generate_task1,
            2: self._generate_task2,
            3: self._generate_task3,
            6: self._generate_task6,
            7: self._generate_task7,
            8: self._generate_task8,
            11: self._generate_task11,
            15: self._generate_task15,
            17: self._generate_task17,
            19: self._generate_task19
        }
        
        if task_id not in generator_map:
            raise ValueError(
                f"Task {task_id} not implemented. "
                f"Available: {list(generator_map.keys())}"
            )
        
        return generator_map[task_id](num_samples)
    
    def _generate_task1(self, num_samples: int) -> List[BabiSample]:
        """Task 1: Single Supporting Fact.
        
        Where is person? -> location (requires 1 fact)
        
        :param num_samples: Number of samples.
        :return: List of samples.
        """
        samples = []
        
        for _ in range(num_samples):
            person = self._rng.choice(self.NAMES)
            locations = self._rng.choice(self.LOCATIONS, size=3, replace=False)
            
            story = []
            for loc in locations[:-1]:
                story.append(f"{person} went to the {loc}")
            
            final_loc = locations[-1]
            supporting_idx = len(story)
            story.append(f"{person} went to the {final_loc}")
            
            # Add distractor
            other_person = self._rng.choice([n for n in self.NAMES if n != person])
            other_loc = self._rng.choice(self.LOCATIONS)
            story.insert(self._rng.integers(0, len(story)), 
                        f"{other_person} went to the {other_loc}")
            
            # Adjust supporting index if distractor was inserted before
            if story.index(f"{person} went to the {final_loc}") != supporting_idx:
                supporting_idx = story.index(f"{person} went to the {final_loc}")
            
            samples.append(BabiSample(
                story=story,
                question=f"Where is {person}?",
                answer=final_loc,
                supporting_facts=[supporting_idx],
                task_id=1
            ))
        
        return samples
    
    def _generate_task2(self, num_samples: int) -> List[BabiSample]:
        """Task 2: Two Supporting Facts.
        
        Where is object? -> location (requires tracking person + object)
        
        :param num_samples: Number of samples.
        :return: List of samples.
        """
        samples = []
        
        for _ in range(num_samples):
            person = self._rng.choice(self.NAMES)
            obj = self._rng.choice(self.OBJECTS)
            loc1 = self._rng.choice(self.LOCATIONS)
            loc2 = self._rng.choice([l for l in self.LOCATIONS if l != loc1])
            
            story = [
                f"{person} picked up the {obj}",
                f"{person} went to the {loc1}",
                f"{person} went to the {loc2}"
            ]
            
            samples.append(BabiSample(
                story=story,
                question=f"Where is the {obj}?",
                answer=loc2,
                supporting_facts=[0, 2],
                task_id=2
            ))
        
        return samples
    
    def _generate_task3(self, num_samples: int) -> List[BabiSample]:
        """Task 3: Three Supporting Facts.
        
        Where was object before location? -> previous location
        
        :param num_samples: Number of samples.
        :return: List of samples.
        """
        samples = []
        
        for _ in range(num_samples):
            person = self._rng.choice(self.NAMES)
            obj = self._rng.choice(self.OBJECTS)
            locs = self._rng.choice(self.LOCATIONS, size=3, replace=False)
            
            story = [
                f"{person} picked up the {obj}",
                f"{person} went to the {locs[0]}",
                f"{person} went to the {locs[1]}",
                f"{person} went to the {locs[2]}"
            ]
            
            samples.append(BabiSample(
                story=story,
                question=f"Where was the {obj} before the {locs[2]}?",
                answer=locs[1],
                supporting_facts=[0, 2, 3],
                task_id=3
            ))
        
        return samples
    
    def _generate_task6(self, num_samples: int) -> List[BabiSample]:
        """Task 6: Yes/No Questions.
        
        Is person in location? -> yes/no
        
        :param num_samples: Number of samples.
        :return: List of samples.
        """
        samples = []
        
        for _ in range(num_samples):
            person = self._rng.choice(self.NAMES)
            actual_loc = self._rng.choice(self.LOCATIONS)
            
            story = [f"{person} went to the {actual_loc}"]
            
            # Add distractors
            for _ in range(self._rng.integers(1, 4)):
                other = self._rng.choice([n for n in self.NAMES if n != person])
                loc = self._rng.choice(self.LOCATIONS)
                story.append(f"{other} went to the {loc}")
            
            self._rng.shuffle(story)
            
            # Randomly ask about actual or different location
            if self._rng.random() > 0.5:
                query_loc = actual_loc
                answer = "yes"
            else:
                query_loc = self._rng.choice([l for l in self.LOCATIONS if l != actual_loc])
                answer = "no"
            
            supporting_idx = next(i for i, s in enumerate(story) if person in s and actual_loc in s)
            
            samples.append(BabiSample(
                story=story,
                question=f"Is {person} in the {query_loc}?",
                answer=answer,
                supporting_facts=[supporting_idx],
                task_id=6
            ))
        
        return samples
    
    def _generate_task7(self, num_samples: int) -> List[BabiSample]:
        """Task 7: Counting.
        
        How many objects is person carrying? -> number
        
        :param num_samples: Number of samples.
        :return: List of samples.
        """
        samples = []
        
        for _ in range(num_samples):
            person = self._rng.choice(self.NAMES)
            num_objects = self._rng.integers(1, 5)
            objects = self._rng.choice(self.OBJECTS, size=num_objects, replace=False)
            
            story = []
            supporting = []
            
            for i, obj in enumerate(objects):
                story.append(f"{person} picked up the {obj}")
                supporting.append(i)
            
            # Maybe drop some
            num_drop = self._rng.integers(0, len(objects))
            dropped = self._rng.choice(objects, size=num_drop, replace=False) if num_drop > 0 else []
            
            for obj in dropped:
                idx = len(story)
                story.append(f"{person} dropped the {obj}")
                supporting.append(idx)
            
            final_count = len(objects) - len(dropped)
            
            samples.append(BabiSample(
                story=story,
                question=f"How many objects is {person} carrying?",
                answer=str(final_count),
                supporting_facts=supporting,
                task_id=7
            ))
        
        return samples
    
    def _generate_task8(self, num_samples: int) -> List[BabiSample]:
        """Task 8: Lists/Sets.
        
        What is person carrying? -> list of objects
        
        :param num_samples: Number of samples.
        :return: List of samples.
        """
        samples = []
        
        for _ in range(num_samples):
            person = self._rng.choice(self.NAMES)
            num_objects = self._rng.integers(1, 4)
            objects = list(self._rng.choice(self.OBJECTS, size=num_objects, replace=False))
            
            story = []
            supporting = []
            
            for i, obj in enumerate(objects):
                story.append(f"{person} picked up the {obj}")
                supporting.append(i)
            
            # Query about one object
            query_obj = self._rng.choice(objects)
            
            samples.append(BabiSample(
                story=story,
                question=f"Is {person} carrying the {query_obj}?",
                answer="yes",
                supporting_facts=supporting,
                task_id=8
            ))
        
        return samples
    
    def _generate_task11(self, num_samples: int) -> List[BabiSample]:
        """Task 11: Basic Coreference.
        
        Resolve pronouns to entities.
        
        :param num_samples: Number of samples.
        :return: List of samples.
        """
        samples = []
        
        for _ in range(num_samples):
            person = self._rng.choice(self.NAMES)
            loc = self._rng.choice(self.LOCATIONS)
            obj = self._rng.choice(self.OBJECTS)
            
            pronoun = "he" if person in ["John", "Daniel", "Bill", "Fred"] else "she"
            
            story = [
                f"{person} went to the {loc}",
                f"{pronoun.capitalize()} picked up the {obj}"
            ]
            
            samples.append(BabiSample(
                story=story,
                question=f"Where is the {obj}?",
                answer=loc,
                supporting_facts=[0, 1],
                task_id=11
            ))
        
        return samples
    
    def _generate_task15(self, num_samples: int) -> List[BabiSample]:
        """Task 15: Basic Deduction.
        
        Apply simple deductive rules.
        
        :param num_samples: Number of samples.
        :return: List of samples.
        """
        samples = []
        
        for _ in range(num_samples):
            animal1 = self._rng.choice(self.ANIMALS)
            animal2 = self._rng.choice([a for a in self.ANIMALS if a != animal1])
            characteristic = self._rng.choice(["afraid of", "bigger than", "faster than"])
            
            target = self._rng.choice(self.ANIMALS)
            while target in [animal1, animal2]:
                target = self._rng.choice(self.ANIMALS)
            
            story = [
                f"{animal1}s are {characteristic} {animal2}s",
                f"{target} is a {animal1}"
            ]
            
            samples.append(BabiSample(
                story=story,
                question=f"Is {target} {characteristic} {animal2}s?",
                answer="yes",
                supporting_facts=[0, 1],
                task_id=15
            ))
        
        return samples
    
    def _generate_task17(self, num_samples: int) -> List[BabiSample]:
        """Task 17: Positional Reasoning.
        
        Spatial relationship reasoning.
        
        :param num_samples: Number of samples.
        :return: List of samples.
        """
        samples = []
        
        for _ in range(num_samples):
            objects = self._rng.choice(self.OBJECTS, size=3, replace=False)
            
            # Create spatial relationships
            story = [
                f"The {objects[0]} is to the left of the {objects[1]}",
                f"The {objects[2]} is to the right of the {objects[1]}"
            ]
            
            samples.append(BabiSample(
                story=story,
                question=f"What is to the left of the {objects[1]}?",
                answer=objects[0],
                supporting_facts=[0],
                task_id=17
            ))
        
        return samples
    
    def _generate_task19(self, num_samples: int) -> List[BabiSample]:
        """Task 19: Path Finding.
        
        Find path between locations.
        
        :param num_samples: Number of samples.
        :return: List of samples.
        """
        samples = []
        
        for _ in range(num_samples):
            locs = list(self._rng.choice(self.LOCATIONS, size=4, replace=False))
            
            # Create path connections
            story = [
                f"The {locs[0]} is north of the {locs[1]}",
                f"The {locs[2]} is east of the {locs[1]}",
                f"The {locs[3]} is south of the {locs[2]}"
            ]
            
            samples.append(BabiSample(
                story=story,
                question=f"How do you go from the {locs[0]} to the {locs[3]}?",
                answer="south, east, south",
                supporting_facts=[0, 1, 2],
                task_id=19
            ))
        
        return samples
    
    def generate_all_tasks(self) -> Dict[int, List[BabiSample]]:
        """Generate samples for all configured tasks.
        
        :return: Dictionary mapping task_id to list of samples.
        """
        results = {}
        for task_id in self.config.task_ids:
            try:
                results[task_id] = self.generate(task_id)
            except ValueError:
                # Skip unimplemented tasks
                continue
        return results
    
    def encode_sample(
        self,
        sample: BabiSample,
        max_story_len: Optional[int] = None,
        max_sentence_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode a bAbI sample as numpy arrays.
        
        :param sample: BabiSample to encode.
        :param max_story_len: Maximum number of sentences.
        :param max_sentence_len: Maximum tokens per sentence.
        :return: Tuple of (story_array, question_array, answer_indices).
        """
        max_story = max_story_len or self.config.max_story_length
        max_sent = max_sentence_len or self.config.max_sentence_length
        
        # Encode story
        story_arr = np.zeros((max_story, max_sent), dtype=np.int32)
        for i, sentence in enumerate(sample.story[:max_story]):
            words = sentence.lower().replace("?", "").replace(".", "").split()
            for j, word in enumerate(words[:max_sent]):
                story_arr[i, j] = self.word2idx.get(word, self.word2idx["<UNK>"])
        
        # Encode question
        question_arr = np.zeros(max_sent, dtype=np.int32)
        q_words = sample.question.lower().replace("?", "").split()
        for j, word in enumerate(q_words[:max_sent]):
            question_arr[j] = self.word2idx.get(word, self.word2idx["<UNK>"])
        
        # Encode answer (as word index)
        answer_words = sample.answer.lower().split()
        answer_idx = self.word2idx.get(answer_words[0], self.word2idx["<UNK>"])
        
        return story_arr, question_arr, np.array([answer_idx], dtype=np.int32)
    
    def encode_batch(
        self,
        samples: List[BabiSample]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode a batch of bAbI samples.
        
        :param samples: List of BabiSample objects.
        :return: Tuple of batched (stories, questions, answers).
        """
        max_story = self.config.max_story_length
        max_sent = self.config.max_sentence_length
        
        stories = np.zeros((len(samples), max_story, max_sent), dtype=np.int32)
        questions = np.zeros((len(samples), max_sent), dtype=np.int32)
        answers = np.zeros((len(samples), 1), dtype=np.int32)
        
        for i, sample in enumerate(samples):
            s, q, a = self.encode_sample(sample, max_story, max_sent)
            stories[i] = s
            questions[i] = q
            answers[i] = a
        
        return stories, questions, answers
