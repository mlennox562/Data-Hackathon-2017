import numpy as np
import pandas as pd

from util import Dataset

for name in ['train', 'test']:
    print("Processing %s..." % name)
    data = pd.read_csv('./input/%s.csv' % name, encoding="ISO-8859-1")

    # Save column names
    if name == 'train':
        # cat_columns = [c for c in data.columns if c.startswith('cat')]
        # num_columns = [c for c in data.columns if c.startswith('cont')]

        num_columns = ['Year', 'Month', 'Total Items', 'Total Quantity', 'BNF Chapter',
                       'BNF Section', 'BNF Paragraph', 'BNF Sub-Paragraph']

        cat_columns = ['VTM_NM', 'VMP_NM', 'AMP_NM', 'Presentation', 'Strength', 'BNF Code']

        Dataset.save_part_features('categorical', cat_columns)
        Dataset.save_part_features('numeric', num_columns)

    Dataset(categorical=data[cat_columns].values).save(name)
    Dataset(numeric=data[num_columns].values.astype(np.float32)).save(name)
    Dataset(id=data['id']).save(name)

    if 'loss' in data.columns:
        Dataset(loss=data['loss']).save(name)

print("Done.")
