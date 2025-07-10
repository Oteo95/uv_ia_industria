"""
Module 1: Modern Industrial AI Workflow - Practical Exercises
Industrial Application of Artificial Intelligence Course

This file contains comprehensive exercises using real SEM particle detection data.
Students will work through a complete MLflow workflow for industrial computer vision.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader 
import json
import yaml
from ultralytics import YOLO
import shutil


# =============================================================================
# EXERCISE 1: MLflow Setup and SEM Dataset Exploration
# =============================================================================

def setup_mlflow_environment(exp_name = "yolo_runs"):
    """
    Set up MLflow tracking environment for SEM particle detection project.
    This exercise demonstrates MLflow configuration for computer vision tasks.
    """
    print("=== Setting up MLflow Environment for SEM Particle Detection ===")
    
    # TODO: Configure MLflow tracking URI
    mlflow_tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # TODO: Set experiment name for SEM particle detection
    experiment_name = exp_name
    
    try:
        # Create or get existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        
        # TODO: Set up experiment tags for organization
        mlflow.set_experiment_tag("project", "industrial_ai_course")
        mlflow.set_experiment_tag("domain", "computer_vision")
        mlflow.set_experiment_tag("data_type", "sem_microscopy")
        mlflow.set_experiment_tag("task", "particle_detection")
        
        return experiment_id
        
    except Exception as e:
        print(f"❌ Failed to setup MLflow environment: {e}")
        return None

def explore_sem_dataset(data_path="data/module_1/sem_particles_split"):
    """
    Explore the SEM particle detection dataset structure and characteristics.
    
    Parameters:
    -----------
    data_path : str
        Path to the SEM dataset directory
        
    Returns:
    --------
    dict
        Dataset statistics and information
    """
    print("=== Exploring SEM Particle Detection Dataset ===")
    
    dataset_info = {
        'train_images': 0,
        'val_images': 0,
        'total_particles': 0,
        'image_formats': set(),
        'image_sizes': [],
        'particles_per_image': []
    }
    
    # TODO: Analyze training set
    train_path = Path(data_path) / "train"
    train_images_path = train_path / "images"
    train_labels_path = train_path / "labels"
    
    if train_images_path.exists():
        # Count training images
        train_images = list(train_images_path.glob("*"))
        dataset_info['train_images'] = len(train_images)
        
        # TODO: Analyze image formats and sizes
        for img_path in train_images[:10]:  # Sample first 10 images
            try:
                if img_path.suffix.lower() in ['.tif', '.png', '.jpg']:
                    dataset_info['image_formats'].add(img_path.suffix.lower())
                    
                    # Load image to get dimensions
                    if img_path.suffix.lower() == '.tif':
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    else:
                        img = cv2.imread(str(img_path))
                    
                    if img is not None:
                        dataset_info['image_sizes'].append(img.shape[:2])
                        
            except Exception as e:
                print(f"⚠️ Could not process {img_path}: {e}")
        
        # TODO: Analyze labels (YOLO format)
        total_particles = 0
        for label_file in train_labels_path.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    particles_count = len(lines)
                    total_particles += particles_count
                    dataset_info['particles_per_image'].append(particles_count)
            except Exception as e:
                print(f"⚠️ Could not read {label_file}: {e}")
        
        dataset_info['total_particles'] = total_particles
    
    # TODO: Analyze validation set
    val_path = Path(data_path) / "val"
    val_images_path = val_path / "images"
    
    if val_images_path.exists():
        val_images = list(val_images_path.glob("*"))
        dataset_info['val_images'] = len(val_images)
    
    # TODO: Print dataset summary
    print(f"📊 Dataset Summary:")
    print(f"   Training images: {dataset_info['train_images']}")
    print(f"   Validation images: {dataset_info['val_images']}")
    print(f"   Total particles detected: {dataset_info['total_particles']}")
    print(f"   Image formats: {list(dataset_info['image_formats'])}")
    
    if dataset_info['particles_per_image']:
        print(f"   Particles per image - Mean: {np.mean(dataset_info['particles_per_image']):.1f}")
        print(f"   Particles per image - Std: {np.std(dataset_info['particles_per_image']):.1f}")
        print(f"   Particles per image - Range: {min(dataset_info['particles_per_image'])}-{max(dataset_info['particles_per_image'])}")
    
    if dataset_info['image_sizes']:
        unique_sizes = list(set(dataset_info['image_sizes']))
        print(f"   Image dimensions: {unique_sizes[:5]}...")  # Show first 5 unique sizes
    
    return dataset_info

def visualize_sem_samples(data_path="data/module_1/sem_particles_split", n_samples=6):
    """
    Visualize sample SEM images with particle annotations.
    
    Parameters:
    -----------
    data_path : str
        Path to the SEM dataset directory
    n_samples : int
        Number of sample images to visualize
    """
    print("=== Visualizing SEM Sample Images ===")
    
    train_images_path = Path(data_path) / "train" / "images"
    train_labels_path = Path(data_path) / "train" / "labels"
    
    # TODO: Get sample images
    image_files = list(train_images_path.glob("*.png"))[:n_samples]  # Use PNG files for better display
    if len(image_files) == 0:
        image_files = list(train_images_path.glob("*.tif"))[:n_samples]  # Fallback to TIF
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, img_path in enumerate(image_files):
        if i >= n_samples:
            break
            
        try:
            # TODO: Load image
            if img_path.suffix.lower() == '.tif':
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # TODO: Load corresponding labels
            label_path = train_labels_path / f"{img_path.stem}.txt"
            particles = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                            particles.append((x_center, y_center, width, height))
            
            # TODO: Draw bounding boxes on image
            h, w = img_rgb.shape[:2]
            for x_center, y_center, width, height in particles:
                # Convert normalized coordinates to pixel coordinates
                x_center_px = int(x_center * w)
                y_center_px = int(y_center * h)
                width_px = int(width * w)
                height_px = int(height * h)
                
                # Calculate bounding box corners
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                
                # Draw rectangle
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # TODO: Display image
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"{img_path.name}\n{len(particles)} particles")
            axes[i].axis('off')
            
        except Exception as e:
            print(f"⚠️ Could not visualize {img_path}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading\n{img_path.name}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(image_files), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle("SEM Particle Detection - Sample Images with Annotations", y=1.02)
    plt.show()


# =============================================================================
# EXERCISE 3: YOLO Model Training with MLflow
# =============================================================================

def prepare_yolo_dataset(data_path="data/module_1/sem_particles_split", output_path="data/module_1/yolo_dataset"):
    """
    Prepare the SEM dataset for YOLO training by creating proper directory structure
    and configuration files.
    
    Parameters:
    -----------
    data_path : str
        Path to the original SEM dataset
    output_path : str
        Path where YOLO-formatted dataset will be created
        
    Returns:
    --------
    str
        Path to the dataset configuration file
    """
    print("=== Preparing YOLO Dataset Structure ===")
    
    # TODO: Create YOLO dataset directory structure
    yolo_path = Path(output_path)
    yolo_path.mkdir(exist_ok=True)
    
    # Create train and val directories
    (yolo_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (yolo_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (yolo_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (yolo_path / "val" / "labels").mkdir(parents=True, exist_ok=True)
    
    # TODO: Copy training data
    src_train_images = Path(data_path) / "train" / "images"
    src_train_labels = Path(data_path) / "train" / "labels"
    dst_train_images = yolo_path / "train" / "images"
    dst_train_labels = yolo_path / "train" / "labels"
    
    if src_train_images.exists():
        for img_file in src_train_images.glob("*"):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']:
                shutil.copy2(img_file, dst_train_images)
        
        for label_file in src_train_labels.glob("*.txt"):
            shutil.copy2(label_file, dst_train_labels)
        
        print(f"✅ Copied {len(list(dst_train_images.glob('*')))} training images")
        print(f"✅ Copied {len(list(dst_train_labels.glob('*')))} training labels")
    
    # TODO: Copy validation data (if exists)
    src_val_images = Path(data_path) / "val" / "images"
    src_val_labels = Path(data_path) / "val" / "labels"
    dst_val_images = yolo_path / "val" / "images"
    dst_val_labels = yolo_path / "val" / "labels"
    
    if src_val_images.exists() and len(list(src_val_images.glob("*"))) > 0:
        for img_file in src_val_images.glob("*"):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']:
                shutil.copy2(img_file, dst_val_images)
        
        if src_val_labels.exists():
            for label_file in src_val_labels.glob("*.txt"):
                shutil.copy2(label_file, dst_val_labels)
        
        print(f"✅ Copied {len(list(dst_val_images.glob('*')))} validation images")
    else:
        # Create validation split from training data if no validation set exists
        print("⚠️ No validation set found, creating split from training data...")
        train_images = list(dst_train_images.glob("*"))
        val_split = int(0.2 * len(train_images))  # 20% for validation
        
        for i, img_file in enumerate(train_images[:val_split]):
            # Move to validation
            val_img_path = dst_val_images / img_file.name
            shutil.move(img_file, val_img_path)
            
            # Move corresponding label
            label_file = dst_train_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                val_label_path = dst_val_labels / label_file.name
                shutil.move(label_file, val_label_path)
        
        print(f"✅ Created validation split with {val_split} images")
    
    # TODO: Create dataset configuration file
    config_data = {
        'path': str(yolo_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,  # number of classes
        'names': ['particle']  # class names
    }
    
    config_path = yolo_path / "dataset.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    print(f"✅ Created dataset configuration: {config_path}")
    
    return str(config_path)

def train_yolo_with_mlflow(dataset_config,experiment="exp1", model_size='n', epochs=3, imgsz=640):
    """
    Train YOLO model on SEM particle detection dataset with MLflow tracking.
    
    Parameters:
    -----------
    dataset_config : str
        Path to the dataset configuration YAML file
    model_size : str
        YOLO model size ('n', 's', 'm', 'l', 'x')
    epochs : int
        Number of training epochs
    imgsz : int
        Image size for training
        
    Returns:
    --------
    str
        MLflow run ID
    """
    print(f"=== Training YOLOv8{model_size} with MLflow Tracking ===")
    
    with mlflow.start_run(run_name=f"YOLOv8{model_size}_SEM_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"Started MLflow run with ID: {run.info.run_id}")
        
        # TODO: Log parameters
        mlflow.log_param("model_type", "YOLOv8")
        mlflow.log_param("model_size", model_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("image_size", imgsz)
        mlflow.log_param("dataset_config", dataset_config)
        mlflow.log_param("task", "object_detection")
        mlflow.log_param("framework", "ultralytics")
        
        try:
            # TODO: Initialize YOLO model
            model = YOLO(f'yolov8{model_size}.pt')  # Load pretrained model
            
            # TODO: Train the model
            print(f"🚀 Starting YOLOv8{model_size} training...")
            results = model.train(
                data=dataset_config,
                epochs=epochs,
                imgsz=imgsz,
                save=True,
                project=experiment,
                name=f"sem_particles_yolov8{model_size}",
                exist_ok=True,
                verbose=True,
            )
            
            # TODO: Log training metrics
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        # Clean metric names to remove invalid characters
                        clean_key = key.replace('(', '_').replace(')', '_').replace('/', '_')
                        mlflow.log_metric(f"final_{clean_key}", value)
            
            # TODO: Get best model path
            best_model_path = results.save_dir / "weights" / "best.pt"
            last_model_path = results.save_dir / "weights" / "last.pt"
            
            # TODO: Log model artifacts
            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path), "model")
                print(f"✅ Logged best model: {best_model_path}")
            
            if last_model_path.exists():
                mlflow.log_artifact(str(last_model_path), "model")
                print(f"✅ Logged last model: {last_model_path}")
            
            # TODO: Log training plots
            plots_dir = results.save_dir
            for plot_file in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(plot_file), "plots")
                print(f"✅ Logged plot: {plot_file.name}")
            
            # TODO: Validate model and log validation metrics
            print("🔍 Running validation...")
            val_results = model.val()
            
            if hasattr(val_results, 'results_dict'):
                val_metrics = val_results.results_dict
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        # Clean metric names to remove invalid characters
                        clean_key = key.replace('(', '_').replace(')', '_').replace('/', '_')
                        mlflow.log_metric(f"val_{clean_key}", value)
            
            # TODO: Log key performance metrics
            if hasattr(val_results, 'box'):
                box_metrics = val_results.box
                mlflow.log_metric("mAP50", box_metrics.map50)
                mlflow.log_metric("mAP50-95", box_metrics.map)
                mlflow.log_metric("precision", box_metrics.p.mean())
                mlflow.log_metric("recall", box_metrics.r.mean())
                
                print(f"📊 Validation Results:")
                print(f"   mAP@0.5: {box_metrics.map50:.4f}")
                print(f"   mAP@0.5:0.95: {box_metrics.map:.4f}")
                print(f"   Precision: {box_metrics.p.mean():.4f}")
                print(f"   Recall: {box_metrics.r.mean():.4f}")
            
            # TODO: Test on sample images and log predictions
            test_images_dir = Path(dataset_config).parent / "val" / "images"
            sample_images = list(test_images_dir.glob("*"))[:5]  # Test on 5 images
            
            if sample_images:
                print("🖼️ Testing on sample images...")
                for i, img_path in enumerate(sample_images):
                    try:
                        # Run inference
                        pred_results = model(str(img_path))
                        
                        # Save prediction image
                        pred_img_path = f"prediction_sample_{i}.jpg"
                        pred_results[0].save(pred_img_path)
                        mlflow.log_artifact(pred_img_path, "predictions")
                        
                        # Log detection count
                        detections = len(pred_results[0].boxes) if pred_results[0].boxes is not None else 0
                        mlflow.log_metric(f"detections_sample_{i}", detections)
                        
                    except Exception as e:
                        print(f"⚠️ Could not process sample {img_path}: {e}")
            
            # TODO: Log model summary
            model_info = {
                "model_type": f"YOLOv8{model_size}",
                "parameters": sum(p.numel() for p in model.model.parameters()),
                "model_size_mb": best_model_path.stat().st_size / (1024*1024) if best_model_path.exists() else 0,
                "training_time": "N/A",  # Could be calculated if needed
                "dataset": "SEM Particle Detection"
            }
            
            # Save model info as JSON
            with open("model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
            mlflow.log_artifact("model_info.json", "model")
            
            run_id = mlflow.active_run().info.run_id
            print(f"✅ YOLOv8{model_size} training completed! Run ID: {run_id}")
            
            return run_id
            
        except Exception as e:
            mlflow.log_param("training_status", "failed")
            mlflow.log_param("error_message", str(e))
            print(f"❌ YOLO training failed: {e}")
            raise e

def compare_yolo_models(dataset_config, model_sizes=['n', 's'], epochs=2):
    """
    Train and compare multiple YOLO model sizes with MLflow tracking.
    
    Parameters:
    -----------
    dataset_config : str
        Path to the dataset configuration YAML file
    model_sizes : list
        List of YOLO model sizes to compare
    epochs : int
        Number of training epochs for each model
        
    Returns:
    --------
    list
        List of MLflow run IDs
    """
    print("=== Comparing YOLO Model Sizes ===")
    
    run_ids = []
    
    for model_size in model_sizes:
        print(f"\n🔄 Training YOLOv8{model_size}...")
        try:
            run_id = train_yolo_with_mlflow(
                dataset_config=dataset_config,
                model_size=model_size,
                epochs=epochs
            )
            run_ids.append(run_id)
            print(f"✅ YOLOv8{model_size} completed with run ID: {run_id}")
        except Exception as e:
            print(f"❌ YOLOv8{model_size} training failed: {e}")
            continue
    
    return run_ids

def register_best_yolo_model(experiment_name="yolo_runs", exp_model="test"):
    """
    Find the best performing YOLO model and register it in MLflow Model Registry.
    
    Parameters:
    -----------
    experiment_name : str
        Name of the MLflow experiment
        
    Returns:
    --------
    tuple
        (model_name, version) of the registered model
    """
    print("=== Registering Best YOLO Model ===")
    
    try:
        # TODO: Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ Experiment {experiment_name} not found")
            return None, None
        
        # TODO: Search for all runs first, then filter for YOLO
        print("🔍 Searching for YOLO runs...")
        all_runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        print(all_runs_df)
        if len(all_runs_df) == 0:
            print("❌ No runs found in experiment")
            return None, None
        
        print(f"📊 Found {len(all_runs_df)} total runs in experiment")
        
        # Filter for YOLO runs manually
        yolo_runs = []
        for idx, run in all_runs_df.iterrows():
            # Check if it's a YOLO run by looking at run name or parameters
            run_name = run.get('tags.mlflow.runName', '')
            model_type = run.get('params.model_type', '')
            yolo_runs.append(run)
        
        if len(yolo_runs) == 0:
            print("❌ No YOLO runs found in experiment")
            print("Available runs:")
            for idx, run in all_runs_df.iterrows():
                run_name = run.get('tags.mlflow.runName', 'Unknown')
                print(f"   - {run_name} (ID: {run.run_id})")
            return None, None
        
        # Find best YOLO run based on available metrics
        best_run = None
        best_score = -1
        
        for run in yolo_runs:
            # Try different metric names that might be available
            score = None
            for metric_name in ['metrics.mAP50', 'metrics.precision', 'metrics.recall']:
                if metric_name in run and pd.notna(run[metric_name]):
                    score = run[metric_name]
                    break
            
            if score is not None and score > best_score:
                best_score = score
                best_run = run
        
        if best_run is None:
            print("❌ No YOLO runs with valid metrics found")
            return None, None
        
        # TODO: Get the best run details
        best_run_id = best_run.run_id
        artifact_uri = best_run.artifact_uri
        best_map50 = best_score  # This is the best score we found
        run_name = best_run.get('tags.mlflow.runName', 'Unknown')
        
        print(f"🏆 Best YOLO model found:")
        print(f"   Run ID: {best_run_id}")
        print(f"   Artifact URI: {artifact_uri}")
        print(f"   Best Score: {best_map50:.4f}")
        print(f"   Model: {run_name}")
        
        # TODO: Register the best model
        model_name = exp_model
        
        # Find the best.pt file in artifacts
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(best_run_id, "model")
        
        best_pt_artifact = None
        for artifact in artifacts:
            if artifact.path.endswith("best.pt"):
                best_pt_artifact = artifact.path
                break
        
        if best_pt_artifact:
            #CHANGE THE URL TO YOU PERSONAL PATH SINCE WE ARE WORKING IN LOCAL
            model_uri = fr"C:\Users\aog13\OneDrive\Escritorio\UV\IA_INDUSTRIA\mlruns\{best_run_id}\artifacts\{best_pt_artifact}"
            registered_model = mlflow.register_model(model_uri, model_name)
            
            # TODO: Transition to Production stage
            client.transition_model_version_stage(
                name=model_name,
                version=registered_model.version,
                stage="Production"
            )
            
            # TODO: Add model description
            client.update_model_version(
                name=model_name,
                version=registered_model.version,
                description=f"YOLOv8 SEM particle detection model with mAP@0.5: {best_map50:.4f}. "
                           f"Trained on industrial SEM microscopy data for automated particle detection."
            )
            
            print(f"✅ YOLO model registered as {model_name} v{registered_model.version} in Production stage")
            
            return model_name, registered_model.version
        else:
            print("❌ Could not find best.pt file in model artifacts")
            return None, None
        
    except Exception as e:
        print(f"❌ YOLO model registration failed: {e}")
        return None, None

# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================

def main_yolo():
    """
    Simplified main function to run only YOLO training workflow.
    Useful for testing YOLO functionality separately.
    """
    print("=== Module 1: YOLO Training Only ===\n")
    
    # Setup
    print("1. Setting up MLflow environment...")
    setup_mlflow_environment()
    
    # Prepare YOLO dataset
    print("\n2. Preparing YOLO dataset...")
    dataset_config = prepare_yolo_dataset()
    
    # Train YOLO models
    print("\n3. Training YOLO models...")
    yolo_run_ids = compare_yolo_models(dataset_config, model_sizes=['n'], epochs=2)
    
    # Register best model
    print("\n4. Registering best YOLO model...")
    yolo_model_name, yolo_version = register_best_yolo_model()

    
    print("\n=== YOLO Training Complete! ===")
    print(f"🎯 Best YOLO model: {yolo_model_name} v{yolo_version}")
    print("Check MLflow UI to see training results!")


if __name__ == "__main__":
    # Uncomment to run all exercises
    #main_yolo()
    
    # For individual exercise testing:
    exp_name="exp0"
    setup_mlflow_environment(exp_name)
    dataset_info = explore_sem_dataset()
    # visualize_sem_samples()
    train_yolo_with_mlflow(r"C:\Users\aog13\OneDrive\Escritorio\UV\IA_INDUSTRIA\data\module_1\yolo_dataset\dataset.yaml", experiment=exp_name)
    register_best_yolo_model(exp_name, "model_test")

    print("Module 1: SEM Particle Detection exercises loaded successfully!")
    print("🔬 Real industrial computer vision dataset ready for MLflow workflow")
    print("\nAvailable functions:")
    print("- setup_mlflow_environment(): Initialize MLflow for SEM project")
    print("- explore_sem_dataset(): Analyze the SEM particle dataset")
    print("- visualize_sem_samples(): View sample SEM images with annotations")
    print("- main_yolo(): Run complete workflow")
