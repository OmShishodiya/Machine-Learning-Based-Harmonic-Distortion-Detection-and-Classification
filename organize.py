import os
import shutil

base_dir = r'd:\UROP\harmonic-distortion-ml'
os.makedirs(base_dir, exist_ok=True)

dirs = [
    'data/raw',
    'data/processed',
    'notebooks',
    'src/preprocessing',
    'src/feature_extraction',
    'src/models',
    'src/filters',
    'src/utils',
    'results/plots',
    'results/metrics',
    'results/logs',
    'scripts',
    'models'
]

for d in dirs:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)

urop_dir = r'd:\UROP'

def move_file(src_name, target_dir, target_name=None):
    src_path = os.path.join(urop_dir, src_name)
    if not target_name: target_name = src_name
    dest_path = os.path.join(base_dir, target_dir, target_name)
    if os.path.exists(src_path):
        print(f'Moving {src_name} to {target_dir}/{target_name}')
        shutil.move(src_path, dest_path)

# Move data
move_file('all_frequencies_1000_1.csv', 'data/raw')
move_file('all_frequencies_1000_2.csv', 'data/raw')
move_file('all_signals_1000_1.csv', 'data/raw')
move_file('all_signals_1000_2.csv', 'data/raw')
move_file('data_info.txt', 'data')

# Plots
move_file('autoencoder_evaluation.png', 'results/plots')
move_file('autoencoder_results.png', 'results/plots')
move_file('nmf_evaluation.png', 'results/plots')
move_file('pca_analysis.png', 'results/plots')
move_file('random_forest_evaluation.png', 'results/plots')
move_file('rf_frequency_evaluation.png', 'results/plots')
move_file('svr_evaluation.png', 'results/plots')
if os.path.exists(os.path.join(urop_dir, 'visualizations')):
    for f in os.listdir(os.path.join(urop_dir, 'visualizations')):
        shutil.move(os.path.join(urop_dir, 'visualizations', f), os.path.join(base_dir, 'results/plots'))
    try: os.rmdir(os.path.join(urop_dir, 'visualizations'))
    except: pass

move_file('error.log', 'results/logs')

# Models
move_file('autoencoder.py', 'src/models')
move_file('cnn_1d_autoencoder.py', 'src/models', 'cnn_autoencoder.py')
move_file('lstm_autoencoder.py', 'src/models')
move_file('one_class_svm.py', 'src/models', 'svm_model.py')
move_file('isolation_forest.py', 'src/models')
move_file('pinn_model.py', 'src/models', 'pinn_model.py') # keep orig name pinn_model.py
move_file('pinn.py', 'src/models', 'pinn.py')
move_file('neural_ode.py', 'src/models')
move_file('fourier_neural_operator.py', 'src/models', 'fno_model.py')
move_file('gan_signal_denoising.py', 'src/models', 'gan_model.py')
move_file('diffusion_model.py', 'src/models')
move_file('sparse_coding.py', 'src/models')
move_file('denoising_autoencoder.py', 'src/models')
move_file('hybrid_fft_autoencoder.py', 'src/models')

# Filters
move_file('kalman_filter.py', 'src/filters')
move_file('wiener_filter.py', 'src/filters')

# Scripts
move_file('train_autoencoder.py', 'scripts', 'run_autoencoder.py')
move_file('train_svr.py', 'scripts', 'run_svm.py')
move_file('train_rf_frequency.py', 'scripts', 'run_rf.py')
move_file('train_nmf.py', 'scripts')
move_file('train_pca.py', 'scripts')
move_file('data_explore.py', 'scripts')

missing_files = {
    'src/preprocessing/data_loader.py': '',
    'src/preprocessing/normalization.py': '',
    'src/feature_extraction/fft_features.py': '',
    'src/feature_extraction/thd_calculation.py': '',
    'src/utils/metrics.py': '',
    'src/utils/visualization.py': '',
    'src/utils/config.py': '',
    'src/models/random_forest.py': '',
    'scripts/run_all_models.py': '',
    'notebooks/eda.ipynb': '',
    'notebooks/feature_analysis.ipynb': '',
    'README.md': '# Harmonic Distortion ML\n',
    '.gitignore': 'venv/\n__pycache__/\n*.pyc\nmodels/*.pkl\nmodels/*.h5\n',
    'LICENSE': 'MIT License\n',
    'setup.py': 'from setuptools import setup, find_packages\nsetup(name=\'harmonic-distortion-ml\', version=\'0.1\', packages=find_packages())\n',
}

for p, content in missing_files.items():
    fp = os.path.join(base_dir, p)
    if not os.path.exists(fp):
        with open(fp, 'w') as f:
            f.write(content)

move_file('requirements.txt', '')

open(os.path.join(base_dir, 'data/sample_dataset.csv'), 'w').close()
open(os.path.join(base_dir, 'models/autoencoder_model.h5'), 'w').close()
open(os.path.join(base_dir, 'models/svm.pkl'), 'w').close()
open(os.path.join(base_dir, 'models/rf.pkl'), 'w').close()

print('Done')
