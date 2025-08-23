"""
Comprehensive examples of what you can build with the spatial layers.

This file demonstrates various architectures and applications enabled by
the spatial layers extracted from the PyTorch CFD codebase.
"""

import keras
from keras import ops

from .perceiver_block import PerceiverBlock
from .continuous_rope import ContinuousRoPE
from .anchor_attention import AnchorAttention
from .supernode_pooling import SupernodePooling
from .perceiver_attention import PerceiverAttention
from .continuous_sin_cos_embed import ContinuousSinCosEmbed
from ..attention.shared_weights_cross_attention import SharedWeightsCrossAttention

# ============================================================================
# 1. COMPUTER VISION & 3D PROCESSING
# ============================================================================

def build_point_cloud_classifier(num_points: int = 2048, num_classes: int = 40):
    """Point Cloud Classification (like ModelNet40).

    Perfect for: 3D object recognition, LIDAR processing, medical 3D imaging.
    """
    inputs = keras.Input(shape=(num_points, 3), name="point_coordinates")

    # Continuous position embedding
    embedded = ContinuousSinCosEmbed(dim=256, ndim=3, name="position_embed")(inputs)

    # Hierarchical attention with supernodes
    attention1 = AnchorAttention(dim=256, num_heads=8, name="attention1")(
        embedded, num_anchor_tokens=512  # 512 anchor points
    )

    attention2 = AnchorAttention(dim=256, num_heads=8, name="attention2")(
        attention1, num_anchor_tokens=128  # Further reduction
    )

    # Global features
    global_features = keras.layers.GlobalMaxPooling1D()(attention2)

    # Classification head
    x = keras.layers.Dense(512, activation="relu")(global_features)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PointCloudClassifier")
    return model


def build_point_cloud_segmentation(num_points: int = 2048, num_classes: int = 13):
    """Point Cloud Segmentation (like ShapeNet parts).

    Perfect for: Part segmentation, semantic segmentation, medical organ segmentation.
    """
    inputs = keras.Input(shape=(num_points, 3), name="point_coordinates")

    # Multi-scale position encoding
    pos_embed_fine = ContinuousSinCosEmbed(dim=128, ndim=3, max_wavelength=1000)(inputs)
    pos_embed_coarse = ContinuousSinCosEmbed(dim=128, ndim=3, max_wavelength=10000)(inputs)
    embedded = keras.layers.Concatenate()([pos_embed_fine, pos_embed_coarse])  # 256-dim

    # Hierarchical processing
    anchors = AnchorAttention(dim=256, num_heads=8)(embedded, num_anchor_tokens=512)

    # Cross-attention back to all points
    perceiver = PerceiverBlock(dim=256, num_heads=8)(
        query_input=embedded,  # All points as queries
        kv_input=anchors  # Anchors as keys/values
    )

    # Point-wise classification
    outputs = keras.layers.Dense(num_classes, activation="softmax")(perceiver)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PointCloudSegmentation")
    return model


def build_3d_scene_understanding():
    """3D Scene Understanding with object detection and relationships.

    Perfect for: Autonomous driving, robotics, AR/VR, indoor navigation.
    """
    # Point cloud input
    points = keras.Input(shape=(None, 3), name="scene_points")

    # Supernode pooling for efficiency
    supernode_indices = keras.Input(shape=(None,), dtype="int32", name="supernode_indices")

    # Extract supernode features
    supernode_features = SupernodePooling(
        hidden_dim=512, ndim=3, radius=2.0, mode="relpos"
    )({
        "positions": points,
        "supernode_indices": supernode_indices
    })

    # Spatial transformer for scene understanding
    scene_features = AnchorAttention(dim=512, num_heads=16)(
        ops.squeeze(supernode_features, 0)  # Remove batch dim
    )

    # Multi-task outputs
    # 1. Object detection
    objectness = keras.layers.Dense(1, activation="sigmoid", name="objectness")(scene_features)
    bbox_regression = keras.layers.Dense(6, name="bbox_coords")(scene_features)  # x,y,z,w,h,d

    # 2. Semantic segmentation
    semantic_logits = keras.layers.Dense(20, activation="softmax", name="semantic")(scene_features)

    # 3. Scene classification
    global_scene = keras.layers.GlobalAveragePooling1D()(scene_features)
    scene_class = keras.layers.Dense(10, activation="softmax", name="scene_class")(global_scene)

    model = keras.Model(
        inputs=[points, supernode_indices],
        outputs={
            "objectness": objectness,
            "bbox_coords": bbox_regression,
            "semantic": semantic_logits,
            "scene_class": scene_class
        },
        name="SceneUnderstanding3D"
    )
    return model


# ============================================================================
# 2. MULTI-MODAL AI APPLICATIONS
# ============================================================================

def build_vision_language_model(max_text_len: int = 77, image_patches: int = 196):
    """Vision-Language Model (like CLIP but with spatial reasoning).

    Perfect for: Image captioning, VQA, cross-modal retrieval, robotics instruction.
    """
    # Vision input (image patches with spatial coordinates)
    patch_features = keras.Input(shape=(image_patches, 768), name="patch_features")
    patch_positions = keras.Input(shape=(image_patches, 2), name="patch_positions")  # 2D grid

    # Text input
    text_tokens = keras.Input(shape=(max_text_len, 512), name="text_tokens")

    # Vision processing with spatial awareness
    visual_pos_embed = ContinuousSinCosEmbed(dim=768, ndim=2)(patch_positions)
    visual_features = patch_features + visual_pos_embed

    # Cross-modal attention
    combined_features = ops.concatenate([visual_features, text_tokens], axis=1)

    # Vision attends to text, text attends to vision
    cross_attended = SharedWeightsCrossAttention(dim=768, num_heads=12)(
        combined_features, split_sizes=[image_patches, max_text_len]
    )

    # Split back
    visual_out = cross_attended[:, :image_patches, :]
    text_out = cross_attended[:, image_patches:, :]

    # Global representations
    visual_global = keras.layers.GlobalAveragePooling1D()(visual_out)
    text_global = keras.layers.GlobalAveragePooling1D()(text_out)

    # Shared embedding space
    visual_embed = keras.layers.Dense(512, name="visual_projection")(visual_global)
    text_embed = keras.layers.Dense(512, name="text_projection")(text_global)

    model = keras.Model(
        inputs=[patch_features, patch_positions, text_tokens],
        outputs={"visual_embed": visual_embed, "text_embed": text_embed},
        name="VisionLanguageModel"
    )
    return model


def build_multi_sensor_fusion():
    """Multi-Sensor Fusion for Autonomous Systems.

    Perfect for: Autonomous vehicles, drones, robots, smart cities.
    """
    # LIDAR point cloud
    lidar_points = keras.Input(shape=(None, 3), name="lidar_points")

    # Camera features (from CNN backbone)
    camera_features = keras.Input(shape=(64, 64, 256), name="camera_features")
    camera_positions = keras.Input(shape=(64, 64, 2), name="camera_positions")  # Pixel coordinates

    # Radar data (sparse)
    radar_points = keras.Input(shape=(None, 4), name="radar_points")  # x,y,z,velocity

    # Process each modality
    # 1. LIDAR processing
    lidar_embed = ContinuousSinCosEmbed(dim=512, ndim=3)(lidar_points)
    lidar_features = AnchorAttention(dim=512, num_heads=8)(lidar_embed, num_anchor_tokens=256)
    lidar_global = keras.layers.GlobalMaxPooling1D()(lidar_features)

    # 2. Camera processing with spatial awareness
    camera_flat = keras.layers.Reshape((64 * 64, 256))(camera_features)
    camera_pos_flat = keras.layers.Reshape((64 * 64, 2))(camera_positions)
    camera_pos_embed = ContinuousSinCosEmbed(dim=256, ndim=2)(camera_pos_flat)
    camera_enhanced = camera_flat + camera_pos_embed
    camera_global = keras.layers.GlobalAveragePooling1D()(camera_enhanced)

    # 3. Radar processing
    radar_embed = ContinuousSinCosEmbed(dim=256, ndim=4)(radar_points)
    radar_global = keras.layers.GlobalMaxPooling1D()(radar_embed)

    # Cross-modal fusion using Perceiver
    all_features = keras.layers.Concatenate()([lidar_global, camera_global, radar_global])

    # Decision outputs
    x = keras.layers.Dense(1024, activation="relu")(all_features)
    x = keras.layers.Dropout(0.3)(x)

    # Multi-task outputs for autonomous driving
    steering = keras.layers.Dense(1, activation="tanh", name="steering")(x)
    acceleration = keras.layers.Dense(1, activation="sigmoid", name="acceleration")(x)
    brake = keras.layers.Dense(1, activation="sigmoid", name="brake")(x)
    hazard_detection = keras.layers.Dense(5, activation="softmax", name="hazards")(x)

    model = keras.Model(
        inputs=[lidar_points, camera_features, camera_positions, radar_points],
        outputs={
            "steering": steering,
            "acceleration": acceleration,
            "brake": brake,
            "hazards": hazard_detection
        },
        name="MultiSensorFusion"
    )
    return model


# ============================================================================
# 3. SCIENTIFIC COMPUTING
# ============================================================================

def build_physics_simulator(grid_size: int = 64):
    """Physics Simulation Network (CFD, Weather, Molecular Dynamics).

    Perfect for: Fluid dynamics, weather prediction, material science, drug discovery.
    """
    # Spatial grid coordinates
    coordinates = keras.Input(shape=(grid_size ** 3, 3), name="grid_coordinates")

    # Initial conditions (pressure, velocity, temperature, etc.)
    initial_state = keras.Input(shape=(grid_size ** 3, 7), name="initial_state")

    # Boundary conditions
    boundary_mask = keras.Input(shape=(grid_size ** 3, 1), name="boundary_mask")

    # Spatial embedding
    pos_embed = ContinuousSinCosEmbed(dim=256, ndim=3)(coordinates)

    # Combine state and position
    state_features = keras.layers.Dense(256)(initial_state)
    combined = pos_embed + state_features

    # Apply boundary conditions
    combined = combined * boundary_mask

    # Physics-aware attention (local interactions)
    physics_features = AnchorAttention(dim=256, num_heads=8)(
        combined, num_anchor_tokens=grid_size ** 2  # Anchor every Z slice
    )

    # Multi-step evolution
    evolved_features = physics_features
    for step in range(5):  # 5 time steps
        evolved_features = AnchorAttention(dim=256, num_heads=8, name=f"evolution_step_{step}")(
            evolved_features
        )

    # Predict next state
    next_state = keras.layers.Dense(7, name="next_state")(evolved_features)

    # Additional physics constraints
    divergence = keras.layers.Dense(1, name="divergence")(evolved_features)  # Should be ~0 for incompressible flow

    model = keras.Model(
        inputs=[coordinates, initial_state, boundary_mask],
        outputs={"next_state": next_state, "divergence": divergence},
        name="PhysicsSimulator"
    )
    return model


def build_molecular_property_predictor():
    """Molecular Property Prediction using spatial molecular graphs.

    Perfect for: Drug discovery, materials science, chemical property prediction.
    """
    # Molecular coordinates
    atom_positions = keras.Input(shape=(None, 3), name="atom_positions")

    # Atom features (atomic number, charge, etc.)
    atom_features = keras.Input(shape=(None, 64), name="atom_features")

    # Bond information
    bond_indices = keras.Input(shape=(None, 2), dtype="int32", name="bond_indices")
    bond_features = keras.Input(shape=(None, 16), name="bond_features")

    # Spatial molecular embedding
    spatial_embed = ContinuousSinCosEmbed(dim=256, ndim=3)(atom_positions)
    atom_embed = keras.layers.Dense(256)(atom_features)

    # Combine spatial and chemical features
    molecular_features = spatial_embed + atom_embed

    # Message passing with spatial awareness
    updated_features = AnchorAttention(dim=256, num_heads=8)(molecular_features)

    # Multiple interaction layers
    for layer in range(3):
        updated_features = AnchorAttention(
            dim=256, num_heads=8, name=f"interaction_layer_{layer}"
        )(updated_features)

    # Graph-level prediction
    molecular_representation = keras.layers.GlobalAttention()(updated_features)

    # Multi-property prediction
    toxicity = keras.layers.Dense(1, activation="sigmoid", name="toxicity")(molecular_representation)
    solubility = keras.layers.Dense(1, name="solubility")(molecular_representation)
    binding_affinity = keras.layers.Dense(1, name="binding_affinity")(molecular_representation)

    model = keras.Model(
        inputs=[atom_positions, atom_features, bond_indices, bond_features],
        outputs={
            "toxicity": toxicity,
            "solubility": solubility,
            "binding_affinity": binding_affinity
        },
        name="MolecularPropertyPredictor"
    )
    return model


# ============================================================================
# 4. ROBOTICS APPLICATIONS
# ============================================================================

def build_manipulation_planner():
    """Robot Manipulation Planning with spatial reasoning.

    Perfect for: Pick-and-place, assembly, dexterous manipulation.
    """
    # Scene point cloud
    scene_points = keras.Input(shape=(None, 3), name="scene_points")

    # Object segmentation mask
    object_mask = keras.Input(shape=(None, 1), name="object_mask")

    # Robot state (joint angles, end-effector pose)
    robot_state = keras.Input(shape=(14,), name="robot_state")  # 7 DOF arm + gripper

    # Goal specification
    goal_position = keras.Input(shape=(3,), name="goal_position")
    goal_orientation = keras.Input(shape=(4,), name="goal_quaternion")

    # Scene understanding
    scene_embed = ContinuousSinCosEmbed(dim=512, ndim=3)(scene_points)
    scene_features = scene_embed * object_mask  # Focus on objects

    # Spatial reasoning for manipulation
    manipulation_features = AnchorAttention(dim=512, num_heads=16)(
        scene_features, num_anchor_tokens=64
    )

    # Global scene understanding
    scene_global = keras.layers.GlobalMaxPooling1D()(manipulation_features)

    # Combine with robot state and goal
    robot_embed = keras.layers.Dense(256)(robot_state)
    goal_embed = keras.layers.Dense(256)(ops.concatenate([goal_position, goal_orientation]))

    combined_state = ops.concatenate([scene_global, robot_embed, goal_embed])

    # Action planning
    x = keras.layers.Dense(1024, activation="relu")(combined_state)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation="relu")(x)

    # Output action (joint velocities)
    joint_actions = keras.layers.Dense(7, activation="tanh", name="joint_actions")(x)
    gripper_action = keras.layers.Dense(1, activation="sigmoid", name="gripper_action")(x)

    # Auxiliary outputs
    success_probability = keras.layers.Dense(1, activation="sigmoid", name="success_prob")(x)

    model = keras.Model(
        inputs=[scene_points, object_mask, robot_state, goal_position, goal_orientation],
        outputs={
            "joint_actions": joint_actions,
            "gripper_action": gripper_action,
            "success_prob": success_probability
        },
        name="ManipulationPlanner"
    )
    return model


def build_slam_system():
    """Simultaneous Localization and Mapping with spatial transformers.

    Perfect for: Mobile robots, autonomous navigation, AR/VR tracking.
    """
    # Current sensor observations
    lidar_scan = keras.Input(shape=(None, 3), name="lidar_scan")

    # Previous map representation
    map_points = keras.Input(shape=(None, 3), name="map_points")
    map_features = keras.Input(shape=(None, 128), name="map_features")

    # Robot odometry
    odometry = keras.Input(shape=(6,), name="odometry")  # x,y,z,roll,pitch,yaw

    # Process current scan
    scan_embed = ContinuousSinCosEmbed(dim=256, ndim=3)(lidar_scan)
    scan_features = AnchorAttention(dim=256, num_heads=8)(scan_embed, num_anchor_tokens=128)

    # Process existing map
    map_embed = ContinuousSinCosEmbed(dim=256, ndim=3)(map_points)
    enhanced_map = map_embed + keras.layers.Dense(256)(map_features)

    # Cross-attention between scan and map for localization
    scan_map_combined = ops.concatenate([scan_features, enhanced_map], axis=1)
    localization_features = SharedWeightsCrossAttention(dim=256, num_heads=8)(
        scan_map_combined,
        split_sizes=[ops.shape(scan_features)[1], ops.shape(enhanced_map)[1]]
    )

    # Extract features for different tasks
    scan_out = localization_features[:, :ops.shape(scan_features)[1], :]
    map_out = localization_features[:, ops.shape(scan_features)[1]:, :]

    # Localization (pose estimation)
    global_features = keras.layers.GlobalAveragePooling1D()(scan_out)
    odometry_features = keras.layers.Dense(128)(odometry)
    pose_features = ops.concatenate([global_features, odometry_features])

    pose_correction = keras.layers.Dense(6, name="pose_correction")(pose_features)

    # Mapping (update map features)
    updated_map_features = keras.layers.Dense(128, name="updated_map_features")(map_out)

    # Loop closure detection
    loop_closure_prob = keras.layers.Dense(1, activation="sigmoid", name="loop_closure")(global_features)

    model = keras.Model(
        inputs=[lidar_scan, map_points, map_features, odometry],
        outputs={
            "pose_correction": pose_correction,
            "updated_map_features": updated_map_features,
            "loop_closure": loop_closure_prob
        },
        name="SLAMSystem"
    )
    return model


# ============================================================================
# 5. GEOSPATIAL & ENVIRONMENTAL AI
# ============================================================================

def build_climate_model():
    """Climate/Weather Prediction Model with spatial transformers.

    Perfect for: Weather forecasting, climate modeling, environmental monitoring.
    """
    # Geographic coordinates (lat, lon, altitude)
    coordinates = keras.Input(shape=(None, 3), name="geo_coordinates")

    # Atmospheric state variables
    temperature = keras.Input(shape=(None, 1), name="temperature")
    pressure = keras.Input(shape=(None, 1), name="pressure")
    humidity = keras.Input(shape=(None, 1), name="humidity")
    wind_velocity = keras.Input(shape=(None, 3), name="wind_velocity")

    # Combine atmospheric variables
    atmospheric_state = ops.concatenate([temperature, pressure, humidity, wind_velocity], axis=-1)

    # Spatial embedding for geographic coordinates
    geo_embed = ContinuousSinCosEmbed(dim=512, ndim=3, max_wavelength=1000000)(coordinates)  # Large scale
    state_embed = keras.layers.Dense(512)(atmospheric_state)

    # Combine geographic and atmospheric information
    climate_features = geo_embed + state_embed

    # Multi-scale atmospheric dynamics
    local_dynamics = AnchorAttention(dim=512, num_heads=8, name="local_dynamics")(
        climate_features, num_anchor_tokens=256
    )

    regional_dynamics = AnchorAttention(dim=512, num_heads=8, name="regional_dynamics")(
        local_dynamics, num_anchor_tokens=64
    )

    # Predict future states
    future_temperature = keras.layers.Dense(1, name="future_temperature")(regional_dynamics)
    future_pressure = keras.layers.Dense(1, name="future_pressure")(regional_dynamics)
    future_humidity = keras.layers.Dense(1, name="future_humidity")(regional_dynamics)
    future_wind = keras.layers.Dense(3, name="future_wind")(regional_dynamics)

    # Extreme weather prediction
    storm_probability = keras.layers.Dense(1, activation="sigmoid", name="storm_prob")(
        keras.layers.GlobalMaxPooling1D()(regional_dynamics)
    )

    model = keras.Model(
        inputs=[coordinates, temperature, pressure, humidity, wind_velocity],
        outputs={
            "future_temperature": future_temperature,
            "future_pressure": future_pressure,
            "future_humidity": future_humidity,
            "future_wind": future_wind,
            "storm_probability": storm_probability
        },
        name="ClimateModel"
    )
    return model


# ============================================================================
# 6. CREATIVE AI & ENTERTAINMENT
# ============================================================================

def build_procedural_world_generator():
    """Procedural World Generation for games and simulations.

    Perfect for: Game development, virtual worlds, architectural design.
    """
    # Seed coordinates for generation
    seed_points = keras.Input(shape=(None, 3), name="seed_points")

    # Control parameters (biome type, elevation, etc.)
    control_params = keras.Input(shape=(None, 8), name="control_parameters")

    # Geographic constraints
    constraints = keras.Input(shape=(None, 4), name="constraints")  # boundaries, water, etc.

    # Spatial embedding for world coordinates
    spatial_embed = ContinuousSinCosEmbed(dim=512, ndim=3, max_wavelength=10000)(seed_points)
    control_embed = keras.layers.Dense(512)(control_params)
    constraint_embed = keras.layers.Dense(512)(constraints)

    # Combine all information
    world_features = spatial_embed + control_embed + constraint_embed

    # Hierarchical world generation
    # Large-scale features (continents, mountain ranges)
    macro_features = AnchorAttention(dim=512, num_heads=16, name="macro_scale")(
        world_features, num_anchor_tokens=64
    )

    # Medium-scale features (cities, forests, rivers)
    meso_features = AnchorAttention(dim=512, num_heads=16, name="meso_scale")(
        macro_features, num_anchor_tokens=256
    )

    # Fine-scale features (buildings, vegetation, roads)
    micro_features = AnchorAttention(dim=512, num_heads=16, name="micro_scale")(
        meso_features
    )

    # Generate different world aspects
    terrain_height = keras.layers.Dense(1, name="terrain_height")(micro_features)
    biome_type = keras.layers.Dense(10, activation="softmax", name="biome_type")(micro_features)
    resource_density = keras.layers.Dense(5, activation="sigmoid", name="resources")(micro_features)
    structure_type = keras.layers.Dense(15, activation="softmax", name="structures")(micro_features)

    # Road/path network
    connectivity = keras.layers.Dense(1, activation="sigmoid", name="connectivity")(micro_features)

    model = keras.Model(
        inputs=[seed_points, control_params, constraints],
        outputs={
            "terrain_height": terrain_height,
            "biome_type": biome_type,
            "resource_density": resource_density,
            "structure_type": structure_type,
            "connectivity": connectivity
        },
        name="ProceduralWorldGenerator"
    )
    return model


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def demonstrate_applications():
    """Demonstrate building and using these applications."""
    print("🚀 Spatial Layers Applications Demo")
    print("=" * 40)

    # 1. Point Cloud Classification
    print("\n📊 1. Point Cloud Classification")
    pc_classifier = build_point_cloud_classifier(num_points=1024, num_classes=10)
    print(f"   Model: {pc_classifier.name}")
    print(f"   Parameters: {pc_classifier.count_params():,}")

    # Test with random data
    test_points = keras.random.uniform((2, 1024, 3)) * 10
    predictions = pc_classifier(test_points)
    print(f"   Input: {test_points.shape} -> Output: {predictions.shape}")

    # 2. Vision-Language Model
    print("\n🎭 2. Vision-Language Model")
    vlm = build_vision_language_model()
    print(f"   Model: {vlm.name}")
    print(f"   Parameters: {vlm.count_params():,}")

    # 3. Physics Simulator
    print("\n🌊 3. Physics Simulator")
    physics_sim = build_physics_simulator(grid_size=32)
    print(f"   Model: {physics_sim.name}")
    print(f"   Parameters: {physics_sim.count_params():,}")

    # 4. Multi-Sensor Fusion
    print("\n🤖 4. Multi-Sensor Fusion")
    sensor_fusion = build_multi_sensor_fusion()
    print(f"   Model: {sensor_fusion.name}")
    print(f"   Parameters: {sensor_fusion.count_params():,}")

    # 5. Climate Model
    print("\n🌍 5. Climate Model")
    climate_model = build_climate_model()
    print(f"   Model: {climate_model.name}")
    print(f"   Parameters: {climate_model.count_params():,}")

    print("\n✨ All models built successfully!")
    print("\n🔑 Key Capabilities Enabled:")
    print("   • Spatial reasoning with continuous coordinates")
    print("   • Hierarchical attention for multi-scale processing")
    print("   • Cross-modal fusion between different data types")
    print("   • Graph-like operations on point clouds")
    print("   • Physics-aware neural networks")
    print("   • Multi-task learning with spatial awareness")


def training_tips():
    """Tips for training models with spatial layers."""
    print("\n📚 Training Tips for Spatial Models:")
    print("=" * 35)

    print("\n1. 📍 Coordinate Normalization:")
    print("   • Normalize coordinates to reasonable ranges (e.g., [0, 1000])")
    print("   • Use consistent coordinate systems across datasets")
    print("   • Consider relative vs absolute positioning")

    print("\n2. 🎯 Attention Strategies:")
    print("   • Start with fewer anchor tokens, increase gradually")
    print("   • Use hierarchical attention for large point clouds")
    print("   • Apply dropout to attention weights for regularization")

    print("\n3. ⚡ Performance Optimization:")
    print("   • Use SupernodePooling for very large point clouds")
    print("   • Batch similar-sized sequences together")
    print("   • Consider mixed precision training")

    print("\n4. 🔄 Data Augmentation:")
    print("   • Random rotations and translations")
    print("   • Point sampling and jittering")
    print("   • Scale variations")

    print("\n5. 📊 Loss Functions:")
    print("   • Chamfer distance for point cloud tasks")
    print("   • Focal loss for imbalanced spatial data")
    print("   • Physics-informed losses for simulations")


if __name__ == "__main__":
    demonstrate_applications()
    training_tips()