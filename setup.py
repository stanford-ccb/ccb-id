from setuptools import setup, find_packages
#from ccbid._version import __version__ as ver
exec(open('ccbid/_version.py').read())

setup_args = {
    'name': 'ccb-id',
    'version': __version__,
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
        'bin/train'
    ],
    'data_files': [
        ('support_files', [
            'ccbid/support_files/gbc.pck',
            'ccbid/support_files/rfc.pck',
            'ccbid/support_files/params-gbc.pck',
            'ccbid/support_files/params-rfc.pck',
            'ccbid/support_files/reducer.pck',
            'ccbid/support_files/training.csv',
            'ccbid/support_files/testing.csv',
            'ccbid/support_files/neon-bands.csv',
            'ccbid/support_files/species_id.csv'
        ])
    ]
}

setup(**setup_args)
