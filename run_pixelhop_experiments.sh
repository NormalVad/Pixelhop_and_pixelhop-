#!/bin/bash
#SBATCH --job-name=pixelhop_exp
#SBATCH --account=ywang234_1595
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=pixelhop_exp_%j.out
#SBATCH --error=pixelhop_exp_%j.err

# Create log directory
mkdir -p /home1/ayushgoy/EE569_hw6_7184517074_Goyal/slurm_logs

# Print node information
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"

# Load modules
module purge
module load conda
module load legacy/CentOS7  
module load gcc/9.2.0
module load cuda/11.2.2

# Set working directory
cd /home1/ayushgoy/EE569_hw6_7184517074_Goyal

# Fix energy_histogram method if it doesn't exist
if ! grep -q "energy_histogram" pixelhop/models/saab_transform.py; then
    echo "Adding energy_histogram method to SaabTransform..."
    cat >> pixelhop/models/saab_transform.py << 'EOL'
    
    def energy_histogram(self):
        """Return energy histogram for visualization"""
        if not hasattr(self, 'energy_ratios'):
            return None
        return self.energy_ratios
EOL
fi

if ! grep -q "energy_histogram" pixelhop/models/cw_saab_transform.py; then
    echo "Adding energy_histogram method to CWSaabTransform..."
    cat >> pixelhop/models/cw_saab_transform.py << 'EOL'
    
    def energy_histogram(self):
        """Return energy histogram for visualization"""
        if not hasattr(self, 'energy_ratios'):
            return None
        return self.energy_ratios
EOL
fi

# Fix the naming conflict in PixelHopUnit
echo "Checking for naming conflict in PixelHopUnit..."
if grep -q "self.transform = transform" pixelhop/models/pixelhop_unit.py; then
    echo "Fixing naming conflict in PixelHopUnit..."
    sed -i 's/self.transform = transform/self.saab_transformer = transform/g' pixelhop/models/pixelhop_unit.py
    sed -i 's/self.transform.fit/self.saab_transformer.fit/g' pixelhop/models/pixelhop_unit.py
    sed -i 's/self.transform.transform/self.saab_transformer.transform/g' pixelhop/models/pixelhop_unit.py
    sed -i 's/self.transform.get_num_parameters/self.saab_transformer.get_num_parameters/g' pixelhop/models/pixelhop_unit.py
    sed -i 's/hasattr(self.transform, /hasattr(self.saab_transformer, /g' pixelhop/models/pixelhop_unit.py
    sed -i 's/self.transform.get_intermediate_nodes/self.saab_transformer.get_intermediate_nodes/g' pixelhop/models/pixelhop_unit.py
    sed -i 's/self.transform.get_discarded_nodes/self.saab_transformer.get_discarded_nodes/g' pixelhop/models/pixelhop_unit.py
    sed -i 's/self.transform.energy_histogram/self.saab_transformer.energy_histogram/g' pixelhop/models/pixelhop_unit.py
fi

# Create and activate conda environment (if needed)
if ! conda env list | grep -q "pixelhop_env"; then
    echo "Creating conda environment pixelhop_env..."
    conda create -y -n pixelhop_env python=3.9
fi
eval "$(conda shell.bash hook)"
conda activate pixelhop_env

# Install requirements
pip install -r requirements.txt

# Set environment variables for GPU usage
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-11.2"

# Test GPU detection
echo "Checking GPU availability for TensorFlow..."
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print('Devices:', tf.config.list_physical_devices()); print('GPU Available:', tf.test.is_gpu_available())" 

# Run experiments
echo "=== Starting PixelHop++ vs PixelHop experiments ==="

# Part (a): PixelHop++ experiments
echo "Running PixelHop++ on full MNIST dataset..."
python main.py --dataset mnist --samples 60000 --th1 0.005 --th2 0.001

# Part (a.3): Try different TH1 values and report test accuracy & model size
echo "Exploring different TH1 values..."
python main.py --dataset mnist --samples 60000 --th2 0.001 --explore_th1

# Part (b): Comparison between PixelHop and PixelHop++
# Since your code has both implementations, we'll run both for comparison
echo "Running comparison between PixelHop and PixelHop++..."
python comparison.py --dataset mnist --samples 60000

# Part (c): Error analysis is already included in the main script
# The confusion matrix is generated automatically

echo "All experiments completed at $(date)"

# Create a backup of results
RESULTS_DIR="/scratch1/ayushgoy/pixelhop_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p ${RESULTS_DIR}
echo "Backing up results to ${RESULTS_DIR}..."
cp -r experiment_* ${RESULTS_DIR}/

# Deactivate conda environment
conda deactivate 