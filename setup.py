from setuptools import setup, find_packages
import ccbid

setup_args = {
    'name': 'ccb-id',
    'version': ccbid.__version__,
    'url': 'https://github.com/stanford-ccb/ccb-id',
    'license': 'MIT',
    'author': 'Christopher Anderson',
    'author_email': 'cbanders@stanford.edu',
    'description': 'Species classification approach using imaging spectroscopy',
    'keywords': [
        'neon',
        'biogeography',
        'classification',
        'machine learning',
        'imaging spectroscopy',
        'remote sensing'
    ],
    'packages': ['ccbid'],
    'include_package_data': True,
    'platforms': 'any',
    'scripts': [
        'bin/train.py'
    ],
    'data_files': [
        ('support_files', [
            'support_files/gbc.pck',
            'support_files/rfc.pck',
            'support_files/params-gbc.pck',
            'support_files/params-rfc.pck',
            'support_files/reducer.pck',
            'support_files/training.csv',
            'support_files/testing.csv',
            'support_files/neon-bands.csv',
            'support_files/species_id.csv'
        ])
    ]
}

setup(**setup_args)
