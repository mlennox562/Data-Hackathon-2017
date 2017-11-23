import numpy as np
import pandas as pd

from util import Dataset

for name in ['train', 'test']:
    print("Processing %s..." % name)
    data = pd.read_csv('./input/%s.csv' % name, header=0, index_col=0)

    num_columns = data.describe().columns.tolist()

    cat_columns = [col for col in data.columns if col not in num_columns]

    for col_type in [num_columns, cat_columns]:
        if 'gp_cost_per_registered_patient' in col_type:
            col_type.remove('gp_cost_per_registered_patient')

        if 'record_id' in col_type:
            col_type.remove('record_id')

    # Save column names
    if name == 'train':

        Dataset.save_part_features('categorical', cat_columns)
        Dataset.save_part_features('numeric', num_columns)

    Dataset(categorical=data[cat_columns].values).save(name)
    Dataset(numeric=data[num_columns].values.astype(np.float32)).save(name)
    Dataset(record_id=data['record_id']).save(name)

    if 'gp_cost_per_registered_patient' in data.columns:
        Dataset(prediction=data['gp_cost_per_registered_patient']).save(name)

print("Done.")
