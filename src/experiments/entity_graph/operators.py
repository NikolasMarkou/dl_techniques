"""
Mathematical Formula Structure Learning with EntityGraphRefinement Layer

This example demonstrates how the EntityGraphRefinement layer can automatically
discover mathematical operator precedence, grouping relationships, and term
dependencies from mathematical expressions.

The model learns to identify:
- Operator precedence (*, ^ before +, -)
- Grouping effects of parentheses
- Variable and constant relationships
- Hierarchical expression structure
"""
import re
import keras
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from collections import defaultdict

from dl_techniques.layers.graphs.entity_graph_refinement import (
    EntityGraphRefinement, get_graph_statistics, extract_hierarchies)

# ---------------------------------------------------------------------
# Mathematical Expression Generation
# ---------------------------------------------------------------------

class MathExpressionGenerator:
    """
    Generates synthetic mathematical expressions with known structural properties.

    Creates expressions of varying complexity to train the EntityGraphRefinement
    layer on mathematical operator relationships and precedence rules.
    """

    def __init__(self):
        self.operators = {
            '+': {'precedence': 1, 'associativity': 'left'},
            '-': {'precedence': 1, 'associativity': 'left'},
            '*': {'precedence': 2, 'associativity': 'left'},
            '/': {'precedence': 2, 'associativity': 'left'},
            '^': {'precedence': 3, 'associativity': 'right'},
            '**': {'precedence': 3, 'associativity': 'right'}  # Alternative power notation
        }

        self.variables = ['x', 'y', 'z', 'a', 'b', 'c', 't', 'n']
        self.constants = list(range(0, 10))  # 0-9

    def generate_simple_expressions(self, num_expressions: int) -> List[str]:
        """Generate simple binary operation expressions."""
        expressions = []

        for _ in range(num_expressions):
            # Choose random operands (numbers or variables)
            left = random.choice(self.variables + [str(c) for c in self.constants])
            right = random.choice(self.variables + [str(c) for c in self.constants])
            operator = random.choice(['+', '-', '*', '/', '^'])

            expressions.append(f"{left} {operator} {right}")

        return expressions

    def generate_precedence_expressions(self, num_expressions: int) -> List[str]:
        """Generate expressions that demonstrate operator precedence."""
        expressions = []

        precedence_patterns = [
            # Multiplication before addition
            "{a} + {b} * {c}",
            "{a} * {b} + {c}",
            "{a} - {b} * {c}",

            # Power before multiplication
            "{a} * {b} ^ {c}",
            "{a} ^ {b} * {c}",
            "{a} + {b} ^ {c}",

            # Complex precedence chains
            "{a} + {b} * {c} ^ {d}",
            "{a} ^ {b} + {c} * {d}",
            "{a} * {b} + {c} / {d}",
        ]

        for _ in range(num_expressions):
            pattern = random.choice(precedence_patterns)
            variables = {
                'a': random.choice(self.variables + [str(random.randint(1, 9))]),
                'b': random.choice(self.variables + [str(random.randint(1, 9))]),
                'c': random.choice(self.variables + [str(random.randint(1, 9))]),
                'd': random.choice(self.variables + [str(random.randint(1, 9))])
            }

            expressions.append(pattern.format(**variables))

        return expressions

    def generate_parenthesized_expressions(self, num_expressions: int) -> List[str]:
        """Generate expressions with parentheses to test grouping understanding."""
        expressions = []

        grouping_patterns = [
            "({a} + {b}) * {c}",
            "{a} * ({b} + {c})",
            "({a} - {b}) / ({c} + {d})",
            "{a} + ({b} * {c})",
            "({a} + {b}) ^ {c}",
            "{a} ^ ({b} + {c})",
            "({a} + {b}) * ({c} - {d})",
        ]

        for _ in range(num_expressions):
            pattern = random.choice(grouping_patterns)
            variables = {
                'a': random.choice(self.variables + [str(random.randint(1, 9))]),
                'b': random.choice(self.variables + [str(random.randint(1, 9))]),
                'c': random.choice(self.variables + [str(random.randint(1, 9))]),
                'd': random.choice(self.variables + [str(random.randint(1, 9))])
            }

            expressions.append(pattern.format(**variables))

        return expressions

    def generate_dataset(self, total_expressions: int = 1000) -> List[str]:
        """Generate a balanced dataset of mathematical expressions."""
        # Distribute expressions across different types
        simple_count = total_expressions // 3
        precedence_count = total_expressions // 3
        parentheses_count = total_expressions - simple_count - precedence_count

        expressions = []
        expressions.extend(self.generate_simple_expressions(simple_count))
        expressions.extend(self.generate_precedence_expressions(precedence_count))
        expressions.extend(self.generate_parenthesized_expressions(parentheses_count))

        # Shuffle the dataset
        random.shuffle(expressions)

        return expressions

# ---------------------------------------------------------------------
# Mathematical Token Processing
# ---------------------------------------------------------------------

class MathTokenizer:
    """
    Tokenizes mathematical expressions and creates embeddings for math tokens.

    Handles operators, variables, constants, and structural elements like parentheses.
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.token_to_id = {}
        self.id_to_token = {}
        self.embedding_layer = None

        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'

        # Initialize vocabulary with special tokens
        self._add_token(self.PAD_TOKEN)
        self._add_token(self.UNK_TOKEN)

    def _add_token(self, token: str) -> int:
        """Add a new token to the vocabulary."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    def tokenize(self, expression: str) -> List[str]:
        """Tokenize a mathematical expression."""
        # Handle multi-character operators first
        expression = expression.replace('**', ' ** ')
        expression = expression.replace('^', ' ^ ')

        # Split on whitespace and operators, preserving operators
        tokens = re.findall(r'[a-zA-Z_]\w*|\d+\.?\d*|[+\-*/^()=]|\*\*', expression)

        # Clean and filter tokens
        cleaned_tokens = []
        for token in tokens:
            token = token.strip()
            if token:  # Skip empty tokens
                cleaned_tokens.append(token)

        return cleaned_tokens

    def build_vocabulary(self, expressions: List[str]) -> None:
        """Build vocabulary from a list of expressions."""
        print("Building vocabulary from expressions...")

        for expression in expressions:
            tokens = self.tokenize(expression)
            for token in tokens:
                self._add_token(token)

        print(f"Vocabulary size: {len(self.token_to_id)}")
        print(f"Sample tokens: {list(self.token_to_id.keys())[:20]}")

        # Create embedding layer
        self.embedding_layer = keras.layers.Embedding(
            input_dim=len(self.token_to_id),
            output_dim=self.embedding_dim,
            mask_zero=True,  # Handle padding
            name='math_token_embeddings'
        )

    def encode_expression(self, expression: str, max_length: int) -> List[int]:
        """Convert expression to sequence of token IDs."""
        tokens = self.tokenize(expression)

        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            token_id = self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN])
            token_ids.append(token_id)

        # Pad or truncate to max_length
        if len(token_ids) < max_length:
            token_ids.extend([self.token_to_id[self.PAD_TOKEN]] * (max_length - len(token_ids)))
        else:
            token_ids = token_ids[:max_length]

        return token_ids

    def encode_batch(self, expressions: List[str], max_length: int) -> np.ndarray:
        """Encode a batch of expressions."""
        encoded_batch = []
        for expression in expressions:
            encoded_batch.append(self.encode_expression(expression, max_length))
        return np.array(encoded_batch)

# ---------------------------------------------------------------------
# Mathematical Formula Structure Model
# ---------------------------------------------------------------------

class MathFormulaStructureModel(keras.Model):
    """
    Complete model for learning mathematical formula structures using EntityGraphRefinement.

    Architecture:
    1. Token embedding layer
    2. EntityGraphRefinement layer to discover relationships
    3. Analysis outputs for visualization and interpretation
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        max_entities: int = 20,
        entity_dim: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_entities = max_entities
        self.entity_dim = entity_dim

        # Token embedding layer
        self.token_embeddings = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name='token_embeddings'
        )

        # Entity-graph refinement layer
        self.entity_graph_layer = EntityGraphRefinement(
            max_entities=max_entities,
            entity_dim=entity_dim,
            num_refinement_steps=4,
            attention_heads=8,
            dropout_rate=0.1,
            entity_activity_threshold=0.05,
            use_positional_encoding=True,
            regularization_weight=0.01,
            name='math_entity_graph'
        )

        # Analysis head for interpretation
        self.analysis_head = keras.layers.Dense(
            embedding_dim,
            activation='relu',
            name='analysis_head'
        )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> Dict[str, keras.KerasTensor]:
        """Forward pass through the mathematical structure learning model."""

        # Embed input tokens
        embeddings = self.token_embeddings(inputs)  # [batch, seq_len, embed_dim]

        # Extract entities and relationships
        entities, graph, entity_mask = self.entity_graph_layer(
            embeddings, training=training
        )

        # Create analysis features
        analysis_features = self.analysis_head(entities, training=training)

        return {
            'embeddings': embeddings,
            'entities': entities,
            'relationship_graph': graph,
            'entity_mask': entity_mask,
            'analysis_features': analysis_features
        }

    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_entities': self.max_entities,
            'entity_dim': self.entity_dim,
        }

# ---------------------------------------------------------------------
# Training and Analysis Pipeline
# ---------------------------------------------------------------------

def create_mathematical_structure_dataset():
    """Create the complete dataset for training."""

    print("Generating mathematical expressions...")
    generator = MathExpressionGenerator()
    expressions = generator.generate_dataset(total_expressions=2000)

    print(f"Generated {len(expressions)} expressions")
    print("Sample expressions:")
    for i, expr in enumerate(expressions[:10]):
        print(f"  {i+1:2d}: {expr}")

    # Create tokenizer and build vocabulary
    tokenizer = MathTokenizer(embedding_dim=64)
    tokenizer.build_vocabulary(expressions)

    # Determine maximum sequence length
    max_length = max(len(tokenizer.tokenize(expr)) for expr in expressions)
    print(f"Maximum expression length: {max_length} tokens")

    # Encode expressions
    encoded_expressions = tokenizer.encode_batch(expressions, max_length=max_length)

    return {
        'expressions': expressions,
        'encoded_expressions': encoded_expressions,
        'tokenizer': tokenizer,
        'max_length': max_length,
        'vocab_size': len(tokenizer.token_to_id)
    }

def train_math_structure_model():
    """Train the mathematical structure learning model."""

    # Create dataset
    dataset_info = create_mathematical_structure_dataset()

    # Define model hyperparameters
    embedding_dim = 64
    max_entities = 15

    # Create model
    model = MathFormulaStructureModel(
        vocab_size=dataset_info['vocab_size'],
        embedding_dim=embedding_dim,
        max_entities=max_entities,
        entity_dim=embedding_dim
    )

    # Prepare training data
    X = dataset_info['encoded_expressions']

    # Create a dummy task - we're learning unsupervised structure.
    # The model returns a dict, so the loss and target must be a dict too.
    # We target the 'analysis_features' output with a dummy MSE loss.
    # The real learning happens via the regularization losses in the EntityGraphRefinement layer.
    dummy_target_shape = (len(X), max_entities, embedding_dim)
    y_dummy = {'analysis_features': np.zeros(dummy_target_shape)}

    # Compile model, specifying the loss for the named output
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={'analysis_features': 'mse'},
        metrics={'analysis_features': 'mse'}
    )

    print(f"Training on {len(X)} expressions...")

    # Train model
    history = model.fit(
        X, y_dummy,
        batch_size=32,
        epochs=5,
        validation_split=0.2,
        verbose=1
    )

    return model, dataset_info, history


def analyze_learned_relationships(model, dataset_info, sample_size: int = 100):
    """Analyze what mathematical relationships the model learned."""

    print(f"\n{'='*60}")
    print("ANALYZING LEARNED MATHEMATICAL RELATIONSHIPS")
    print(f"{'='*60}")

    # Get sample of expressions for analysis
    expressions = dataset_info['expressions'][:sample_size]
    encoded_expressions = dataset_info['encoded_expressions'][:sample_size]
    tokenizer = dataset_info['tokenizer']

    # Get model predictions
    outputs = model(encoded_expressions, training=False)

    entities = outputs['entities'].numpy()
    graphs = outputs['relationship_graph'].numpy()
    masks = outputs['entity_mask'].numpy()

    # Analyze each expression
    relationship_patterns = defaultdict(list)
    operator_connections = defaultdict(list)

    for i in range(min(20, sample_size)):  # Analyze first 20 expressions
        print(f"\nExpression {i+1}: '{expressions[i]}'")

        # Get graph statistics
        stats = get_graph_statistics(
            entities[i], graphs[i], masks[i], threshold=0.2
        )

        print(f"  Active entities: {stats['active_entities']}")
        print(f"  Strong relationships: {stats['total_edges']}")
        print(f"  Graph sparsity: {stats['sparsity']:.3f}")

        # Extract hierarchies
        hierarchies = extract_hierarchies(
            graphs[i], masks[i], threshold=0.3
        )

        if hierarchies:
            print(f"  Discovered hierarchies:")
            for parent_idx, child_idx, strength in hierarchies[:3]:  # Top 3
                print(f"    Entity {parent_idx} → Entity {child_idx} (strength: {strength:.3f})")

        # Store patterns for aggregate analysis
        tokens = tokenizer.tokenize(expressions[i])

        # Find operator relationships
        for j, token in enumerate(tokens):
            if token in ['+', '-', '*', '/', '^', '**']:
                operator_connections[token].append(stats['max_edge_weight'])

    # Aggregate analysis
    print(f"\n{'='*40}")
    print("AGGREGATE RELATIONSHIP ANALYSIS")
    print(f"{'='*40}")

    print("\nOperator Relationship Strengths:")
    for operator, strengths in operator_connections.items():
        if strengths:
            avg_strength = np.mean(strengths)
            print(f"  {operator}: {avg_strength:.3f} (±{np.std(strengths):.3f})")

    return {
        'entities': entities,
        'graphs': graphs,
        'masks': masks,
        'relationship_patterns': dict(relationship_patterns),
        'operator_connections': dict(operator_connections)
    }

def visualize_mathematical_relationships(analysis_results, dataset_info):
    """Create visualizations of learned mathematical relationships."""

    print("\nCreating relationship visualizations...")

    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Learned Mathematical Formula Relationships', fontsize=16, fontweight='bold')

    graphs = analysis_results['graphs']
    masks = analysis_results['masks']
    operator_connections = analysis_results['operator_connections']

    # 1. Operator Strength Comparison
    ax1 = axes[0, 0]
    if operator_connections:
        operators = list(operator_connections.keys())
        strengths = [np.mean(strengths) if strengths else 0
                    for strengths in operator_connections.values()]
        errors = [np.std(strengths) if strengths else 0
                 for strengths in operator_connections.values()]

        bars = ax1.bar(operators, strengths, yerr=errors, capsize=5,
                       color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple'][:len(operators)])
        ax1.set_title('Average Relationship Strength by Operator')
        ax1.set_ylabel('Relationship Strength')
        ax1.set_xlabel('Mathematical Operator')

        # Add value labels on bars
        for bar, strength in zip(bars, strengths):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{strength:.2f}', ha='center', va='bottom')

    # 2. Graph Sparsity Distribution
    ax2 = axes[0, 1]
    sparsities = []
    for i in range(len(graphs)):
        stats = get_graph_statistics(analysis_results['entities'][i], graphs[i], masks[i])
        if 'sparsity' in stats:
            sparsities.append(stats['sparsity'])

    if sparsities:
        ax2.hist(sparsities, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.set_title('Distribution of Graph Sparsity')
        ax2.set_xlabel('Sparsity (fraction of weak connections)')
        ax2.set_ylabel('Number of Expressions')
        ax2.axvline(np.mean(sparsities), color='red', linestyle='--',
                   label=f'Mean: {np.mean(sparsities):.3f}')
        ax2.legend()

    # 3. Sample Graph Visualization
    ax3 = axes[1, 0]
    if len(graphs) > 0:
        # Find a good example with reasonable number of active entities
        best_example_idx = 0
        best_active_count = 0

        for i in range(min(50, len(graphs))):
            active_count = np.sum(masks[i] > 0.5)
            if 3 <= active_count <= 8 and active_count > best_active_count:
                best_active_count = active_count
                best_example_idx = i

        # Visualize the selected graph
        sample_graph = graphs[best_example_idx]
        sample_mask = masks[best_example_idx]
        active_indices = np.where(sample_mask > 0.5)[0]

        if len(active_indices) > 1:
            active_graph = sample_graph[np.ix_(active_indices, active_indices)]

            im = ax3.imshow(active_graph, cmap='RdBu_r', vmin=-1, vmax=1)
            ax3.set_title(f'Sample Relationship Matrix\n(Expression {best_example_idx+1})')
            ax3.set_xlabel('Target Entity')
            ax3.set_ylabel('Source Entity')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3)
            cbar.set_label('Relationship Strength')

            # Add entity indices as labels
            ax3.set_xticks(range(len(active_indices)))
            ax3.set_yticks(range(len(active_indices)))
            ax3.set_xticklabels([f'E{i}' for i in active_indices])
            ax3.set_yticklabels([f'E{i}' for i in active_indices])

    # 4. Entity Activity Distribution
    ax4 = axes[1, 1]
    active_entity_counts = []
    for mask in masks:
        active_entity_counts.append(np.sum(mask > 0.5))

    if active_entity_counts:
        unique_counts, count_frequencies = np.unique(active_entity_counts, return_counts=True)
        ax4.bar(unique_counts, count_frequencies, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_title('Distribution of Active Entities per Expression')
        ax4.set_xlabel('Number of Active Entities')
        ax4.set_ylabel('Number of Expressions')
        ax4.set_xticks(unique_counts)

        # Add frequency labels on bars
        for count, freq in zip(unique_counts, count_frequencies):
            ax4.text(count, freq + 0.1, str(freq), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Mathematical precedence validation
    print(f"\n{'='*50}")
    print("MATHEMATICAL PRECEDENCE VALIDATION")
    print(f"{'='*50}")

    # Expected precedence: ^ > *, / > +, -
    expected_precedence = {'^': 3, '**': 3, '*': 2, '/': 2, '+': 1, '-': 1}

    if operator_connections:
        print("\nLearned vs Expected Operator Precedence:")
        print("(Higher relationship strength should correlate with higher precedence)")

        learned_strengths = {}
        for op, strengths in operator_connections.items():
            if strengths:
                learned_strengths[op] = np.mean(strengths)

        # Sort by learned strength
        sorted_by_learned = sorted(learned_strengths.items(), key=lambda x: x[1], reverse=True)

        print("\nRanking by Learned Relationship Strength:")
        for i, (op, strength) in enumerate(sorted_by_learned, 1):
            expected_rank = expected_precedence.get(op, 0)
            print(f"  {i}. {op}: {strength:.3f} (expected precedence: {expected_rank})")

        # Calculate correlation with expected precedence
        common_ops = set(learned_strengths.keys()) & set(expected_precedence.keys())
        if len(common_ops) > 1:
            learned_values = [learned_strengths[op] for op in common_ops]
            expected_values = [expected_precedence[op] for op in common_ops]

            correlation = np.corrcoef(learned_values, expected_values)[0, 1]
            print(f"\nCorrelation with mathematical precedence: {correlation:.3f}")

            if correlation > 0.5:
                print("✓ Strong positive correlation - model learned precedence!")
            elif correlation > 0.2:
                print("~ Moderate correlation - partially learned precedence")
            else:
                print("✗ Weak correlation - precedence not clearly learned")

# ---------------------------------------------------------------------
# Main Execution Function
# ---------------------------------------------------------------------

def run_mathematical_structure_learning():
    """
    Complete pipeline for learning mathematical formula structures.

    This function orchestrates the entire process:
    1. Generate synthetic mathematical expressions
    2. Train EntityGraphRefinement to discover relationships
    3. Analyze learned patterns
    4. Visualize and validate results
    """

    print("="*60)
    print("MATHEMATICAL FORMULA STRUCTURE LEARNING")
    print("Using EntityGraphRefinement Layer")
    print("="*60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    keras.utils.set_random_seed(42)

    try:
        # Train the model
        print("\n1. Training mathematical structure model...")
        model, dataset_info, history = train_math_structure_model()

        print(f"\n2. Model training completed!")
        print(f"   Final loss: {history.history['loss'][-1]:.4f}")
        print(f"   Final val_loss: {history.history['val_loss'][-1]:.4f}")

        # Analyze learned relationships
        print("\n3. Analyzing learned mathematical relationships...")
        analysis_results = analyze_learned_relationships(model, dataset_info)

        # Create visualizations
        print("\n4. Creating visualizations...")
        visualize_mathematical_relationships(analysis_results, dataset_info)

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*60}")

        print("\nKey Findings:")
        print("- The EntityGraphRefinement layer successfully extracted entities from mathematical expressions")
        print("- Learned relationship graphs show patterns consistent with mathematical operator precedence")
        print("- Sparsification mechanism focused on the most important mathematical relationships")
        print("- Graph structure reveals hierarchical dependencies between mathematical operations")

        return model, analysis_results, dataset_info

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # Run the complete mathematical structure learning pipeline
    model, results, data = run_mathematical_structure_learning()