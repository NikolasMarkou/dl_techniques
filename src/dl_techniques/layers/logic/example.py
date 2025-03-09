"""
Example of using Logical Neural Networks for the friends-smokers problem.

This example demonstrates how to use LNNs to implement a classic problem in
statistical relational learning:
- People who smoke are more likely to get cancer
- Friends of smokers are more likely to smoke
"""

import keras
import numpy as np
import matplotlib.pyplot as plt

from .advanced_gates import (
    LogicSystem,
    FuzzyANDGateLayer,
    FuzzyORGateLayer,
    FuzzyNOTGateLayer,
    FuzzyImpliesGateLayer,
)

# ---------------------------------------------------------------------

class FriendsSmokersProblem:
    """
    Implementation of the friends-smokers problem using Logical Neural Networks.

    This class demonstrates how to use Logical Neural Networks to implement
    a classic problem in statistical relational learning, where:
    - People who smoke are more likely to get cancer
    - Friends of smokers are more likely to smoke
    """

    def __init__(
            self,
            logic_system: LogicSystem = LogicSystem.LUKASIEWICZ,
            use_bounds: bool = True,
            temperature: float = 0.1,
    ):
        """
        Initialize the friends-smokers problem.

        Args:
            logic_system: Logic system to use
            use_bounds: Whether to use truth bounds
            temperature: Temperature parameter for logical operations
        """
        self.logic_system = logic_system
        self.use_bounds = use_bounds
        self.temperature = temperature

        # People in our model
        self.people = ["Alice", "Bob", "Charlie", "Dave"]

        # Build the model
        self.model = self._build_model()

        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

    def _build_model(self) -> keras.Model:
        """
        Build the logical neural network for the friends-smokers problem.

        Returns:
            Keras model implementing the friends-smokers problem
        """
        # Input layer for all our propositions
        # Format: [Smokes(Alice), Smokes(Bob), Smokes(Charlie), Smokes(Dave),
        #          Cancer(Alice), Cancer(Bob), Cancer(Charlie), Cancer(Dave),
        #          Friends(Alice,Bob), Friends(Bob,Charlie), Friends(Charlie,Dave)]
        inputs = keras.Input(shape=(11,), name="propositions")

        # Split inputs into individual propositions
        smokes = {}
        cancer = {}
        friends = {}

        for i, person in enumerate(self.people):
            # Extract smoking status for each person
            smokes[person] = keras.layers.Lambda(
                lambda x: x[:, i:i + 1],
                name=f"Smokes_{person}"
            )(inputs)

            # Extract cancer status for each person
            cancer[person] = keras.layers.Lambda(
                lambda x: x[:, i + 4:i + 5],
                name=f"Cancer_{person}"
            )(inputs)

        # Extract friendship relationships
        friends["Alice_Bob"] = keras.layers.Lambda(
            lambda x: x[:, 8:9],
            name="Friends_Alice_Bob"
        )(inputs)
        friends["Bob_Charlie"] = keras.layers.Lambda(
            lambda x: x[:, 9:10],
            name="Friends_Bob_Charlie"
        )(inputs)
        friends["Charlie_Dave"] = keras.layers.Lambda(
            lambda x: x[:, 10:11],
            name="Friends_Charlie_Dave"
        )(inputs)

        # Rule 1: Smoking causes cancer (with some probability)
        smoking_cancer_rules = {}
        for person in self.people:
            # Create Smokes(X) => Cancer(X) for each person
            smoking_cancer_rules[person] = FuzzyImpliesGateLayer(
                logic_system=self.logic_system,
                use_bounds=self.use_bounds,
                trainable_weights=True,
                initial_weight=0.8,
                temperature=self.temperature,
                name=f"SmokesImpliesCancer_{person}"
            )([smokes[person], cancer[person]])

        # Rule 2: Friends of smokers are likely to smoke
        friendship_rules = {}
        friendship_pairs = [
            ("Alice", "Bob"),
            ("Bob", "Charlie"),
            ("Charlie", "Dave")
        ]

        for person1, person2 in friendship_pairs:
            # Create Friends(X,Y) AND Smokes(X) => Smokes(Y)
            # First create AND node for Friends(X,Y) AND Smokes(X)
            friendship_key = f"{person1}_{person2}"
            and_node = FuzzyANDGateLayer(
                logic_system=self.logic_system,
                use_bounds=self.use_bounds,
                temperature=self.temperature,
                name=f"FriendsAndSmokes_{person1}_{person2}"
            )([friends[friendship_key], smokes[person1]])

            # Then create implication node for (Friends(X,Y) AND Smokes(X)) => Smokes(Y)
            friendship_rules[friendship_key] = FuzzyImpliesGateLayer(
                logic_system=self.logic_system,
                use_bounds=self.use_bounds,
                trainable_weights=True,
                initial_weight=0.7,
                temperature=self.temperature,
                name=f"FriendsSmokeImpliesSmoke_{person1}_{person2}"
            )([and_node, smokes[person2]])

        # Collect all rule outputs
        rule_outputs = []
        for rule in smoking_cancer_rules.values():
            rule_outputs.append(rule)
        for rule in friendship_rules.values():
            rule_outputs.append(rule)

        # Create a combined model with all rules
        return keras.Model(inputs=inputs, outputs=rule_outputs)

    def evaluate_formula(self, evidence: dict) -> dict:
        """
        Evaluate the logical formula with given evidence.

        Args:
            evidence: Dictionary mapping proposition names to truth values

        Returns:
            Dictionary of inferred truth values for all propositions
        """
        # Create input tensor from evidence
        input_data = np.zeros((1, 11))

        # Set evidence for smoking
        for i, person in enumerate(self.people):
            if f"Smokes({person})" in evidence:
                input_data[0, i] = evidence[f"Smokes({person})"]
            else:
                input_data[0, i] = 0.5  # Unknown

        # Set evidence for cancer
        for i, person in enumerate(self.people):
            if f"Cancer({person})" in evidence:
                input_data[0, i + 4] = evidence[f"Cancer({person})"]
            else:
                input_data[0, i + 4] = 0.5  # Unknown

        # Set evidence for friendship
        friendship_pairs = [
            ("Alice", "Bob"),
            ("Bob", "Charlie"),
            ("Charlie", "Dave")
        ]
        for i, (person1, person2) in enumerate(friendship_pairs):
            if f"Friends({person1},{person2})" in evidence:
                input_data[0, i + 8] = evidence[f"Friends({person1},{person2})"]
            else:
                input_data[0, i + 8] = 0.5  # Unknown

        # Run forward pass
        predictions = self.model.predict(input_data)

        # Extract rule satisfaction levels
        rule_satisfaction = {}
        for i, person in enumerate(self.people):
            rule_satisfaction[f"SmokesImpliesCancer({person})"] = predictions[i][0][0]

        friendship_pairs = [
            ("Alice", "Bob"),
            ("Bob", "Charlie"),
            ("Charlie", "Dave")
        ]
        for i, (person1, person2) in enumerate(friendship_pairs):
            rule_key = f"FriendsSmokeImpliesSmoke({person1},{person2})"
            rule_satisfaction[rule_key] = predictions[i + 4][0][0]

        # Return input data and rule satisfaction
        inferred = {}

        # Return smoking status
        for i, person in enumerate(self.people):
            inferred[f"Smokes({person})"] = input_data[0, i]

        # Return cancer status
        for i, person in enumerate(self.people):
            inferred[f"Cancer({person})"] = input_data[0, i + 4]

        # Return friendship status
        friendship_pairs = [
            ("Alice", "Bob"),
            ("Bob", "Charlie"),
            ("Charlie", "Dave")
        ]
        for i, (person1, person2) in enumerate(friendship_pairs):
            inferred[f"Friends({person1},{person2})"] = input_data[0, i + 8]

        # Add rule satisfaction
        inferred.update(rule_satisfaction)

        return inferred

    def infer(self, evidence: dict, iterations: int = 5) -> dict:
        """
        Perform iterative inference to update beliefs based on evidence.

        This is a simple form of belief propagation where we iteratively:
        1. Update proposition beliefs based on rule satisfaction
        2. Re-evaluate rules based on updated propositions

        Args:
            evidence: Dictionary mapping proposition names to truth values
            iterations: Number of inference iterations to perform

        Returns:
            Dictionary of inferred truth values for all propositions
        """
        # Start with the evidence
        current_beliefs = evidence.copy()

        # Initialize beliefs for missing propositions
        for person in self.people:
            if f"Smokes({person})" not in current_beliefs:
                current_beliefs[f"Smokes({person})"] = 0.5
            if f"Cancer({person})" not in current_beliefs:
                current_beliefs[f"Cancer({person})"] = 0.5

        friendship_pairs = [
            ("Alice", "Bob"),
            ("Bob", "Charlie"),
            ("Charlie", "Dave")
        ]
        for person1, person2 in friendship_pairs:
            if f"Friends({person1},{person2})" not in current_beliefs:
                current_beliefs[f"Friends({person1},{person2})"] = 0.5

        # Perform iterative inference
        results_history = [current_beliefs.copy()]

        for _ in range(iterations):
            # Evaluate formula with current beliefs
            inferred = self.evaluate_formula(current_beliefs)

            # Update beliefs based on rules
            for person in self.people:
                # Update cancer belief based on smoking rule
                if f"Cancer({person})" not in evidence:
                    smoking_prob = current_beliefs[f"Smokes({person})"]
                    rule_satisfaction = inferred[f"SmokesImpliesCancer({person})"]

                    # If person smokes and rule is satisfied, increase cancer belief
                    if smoking_prob > 0.5 and rule_satisfaction > 0.5:
                        current_beliefs[f"Cancer({person})"] = min(
                            0.95,
                            current_beliefs[f"Cancer({person})"] + 0.1 * rule_satisfaction
                        )

            # Update smoking belief based on friends rules
            for person1, person2 in friendship_pairs:
                rule_key = f"FriendsSmokeImpliesSmoke({person1},{person2})"
                rule_satisfaction = inferred[rule_key]

                if f"Smokes({person2})" not in evidence:
                    friend_smokes = current_beliefs[f"Smokes({person1})"]
                    is_friend = current_beliefs[f"Friends({person1},{person2})"]

                    # If friend smokes and they are friends and rule is satisfied, increase smoking belief
                    if friend_smokes > 0.5 and is_friend > 0.5 and rule_satisfaction > 0.5:
                        current_beliefs[f"Smokes({person2})"] = min(
                            0.95,
                            current_beliefs[f"Smokes({person2})"] + 0.1 * rule_satisfaction
                        )

            # Save history
            results_history.append(current_beliefs.copy())

        return current_beliefs, results_history

    def visualize_inference(self, evidence: dict, iterations: int = 5):
        """
        Visualize the inference process.

        Args:
            evidence: Dictionary mapping proposition names to truth values
            iterations: Number of inference iterations to perform
        """
        # Run inference
        final_beliefs, history = self.infer(evidence, iterations)

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot smoking beliefs
        plt.subplot(2, 1, 1)
        for person in self.people:
            values = [h[f"Smokes({person})"] for h in history]
            plt.plot(values, marker='o', label=f"Smokes({person})")

        plt.xlabel('Iteration')
        plt.ylabel('Belief (0-1)')
        plt.title('Smoking Beliefs Over Time')
        plt.legend()
        plt.grid(True)

        # Plot cancer beliefs
        plt.subplot(2, 1, 2)
        for person in self.people:
            values = [h[f"Cancer({person})"] for h in history]
            plt.plot(values, marker='o', label=f"Cancer({person})")

        plt.xlabel('Iteration')
        plt.ylabel('Belief (0-1)')
        plt.title('Cancer Beliefs Over Time')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# ---------------------------------------------------------------------


def run_example():
    """Run a simple example of the friends-smokers problem."""
    # Create the model
    problem = FriendsSmokersProblem(logic_system=LogicSystem.LUKASIEWICZ)

    # Define evidence
    evidence = {
        "Smokes(Alice)": 0.9,  # Alice is likely a smoker
        "Cancer(Dave)": 0.8,  # Dave likely has cancer
        "Friends(Alice,Bob)": 1.0,  # Alice and Bob are definitely friends
        "Friends(Bob,Charlie)": 1.0,  # Bob and Charlie are definitely friends
        "Friends(Charlie,Dave)": 1.0,  # Charlie and Dave are definitely friends
    }

    # Run inference
    inferred, history = problem.infer(evidence, iterations=10)

    # Print results
    print("Initial evidence:")
    for prop, value in evidence.items():
        print(f"  - {prop}: {value:.2f}")

    print("\nInferred results:")
    for person in problem.people:
        smoke_val = inferred[f"Smokes({person})"]
        cancer_val = inferred[f"Cancer({person})"]
        smoke_str = "High" if smoke_val > 0.7 else "Medium" if smoke_val > 0.4 else "Low"
        cancer_str = "High" if cancer_val > 0.7 else "Medium" if cancer_val > 0.4 else "Low"

        print(f"  - {person}: Smoking probability: {smoke_val:.2f} ({smoke_str}), "
              f"Cancer probability: {cancer_val:.2f} ({cancer_str})")

    # Visualize inference
    problem.visualize_inference(evidence)

    return problem, inferred, history

# ---------------------------------------------------------------------


if __name__ == "__main__":
    problem, inferred, history = run_example()