import yaml
import subprocess

def load_environment_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_installed_packages():
    result = subprocess.run(['conda', 'list'], stdout=subprocess.PIPE)
    installed_packages = result.stdout.decode('utf-8').split('\n')
    installed_packages = [line.split()[0] for line in installed_packages if line]
    return installed_packages

def verify_packages(env_file_path):
    env_data = load_environment_file(env_file_path)
    dependencies = env_data.get('dependencies', [])
    conda_packages = [dep for dep in dependencies if isinstance(dep, str)]
    pip_packages = []
    for dep in dependencies:
        if isinstance(dep, dict) and 'pip' in dep:
            pip_packages.extend(dep['pip'])

    installed_packages = get_installed_packages()

    missing_packages = [pkg for pkg in conda_packages + pip_packages if pkg not in installed_packages]

    if missing_packages:
        print("Missing packages:", missing_packages)
    else:
        print("All packages are installed.")

if __name__ == "__main__":
    verify_packages('environment.yml')