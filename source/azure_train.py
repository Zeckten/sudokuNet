from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import os

def setup_azure_compute(ws, compute_name, vm_size):
    """Setup Azure compute target"""
    try:
        compute_target = ComputeTarget(workspace=ws, name=compute_name)
        print('Found existing compute target')
    except ComputeTargetException:
        print('Creating new compute target...')
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size,
            min_nodes=0,
            max_nodes=4,
            idle_seconds_before_scaledown=1800
        )
        compute_target = ComputeTarget.create(ws, compute_name, compute_config)
        compute_target.wait_for_completion(show_output=True)
    return compute_target

def main():
    # Connect to Azure workspace
    ws = Workspace.from_config()
    
    # Setup compute target
    compute_target = setup_azure_compute(ws, 'gpu-cluster', 'STANDARD_D1')
    
    # Create Azure ML environment
    env = Environment.from_pip_requirements('sudoku-env', 'requirements.txt')
    env.docker.enabled = True
    env.docker.base_image = 'pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime'
    
    # Create experiment
    experiment = Experiment(workspace=ws, name='sudoku-solver-training')
    
    # Configure the training run
    src = ScriptRunConfig(
        source_directory='./source',
        script='SudokuSolver.py',
        compute_target=compute_target,
        environment=env,
        arguments=[
            '--data_path', ws.datasets.get('sudoku'),
            '--batch_size', 1024,
            '--epochs', 20,
            '--sample_size', 100000
        ]
    )
    
    # Submit experiment
    run = experiment.submit(src)
    run.wait_for_completion(show_output=True)

if __name__ == "__main__":
    main() 