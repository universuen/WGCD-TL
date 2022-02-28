import pandas as pd

import src.config.path_config
from scripts.ablation_test import highlight_legal_cells
from scripts.applicability_test import highlight_higher_cells
from scripts.datasets import DATASETS

if __name__ == '__main__':
    src_path = src.config.path_config.test_results / 'ablation_2-15.xlsx'
    dst_path = src.config.path_config.test_results / 'ablation_final.xlsx'
    with pd.ExcelWriter(dst_path) as writer:
        for metric_name in ['F1', 'AUC', 'G-Mean']:
            df = pd.read_excel(src_path, index_col=0, sheet_name=metric_name).loc[DATASETS]
            df.to_excel(writer, metric_name)
            df.style.apply(highlight_legal_cells, axis=1).to_excel(writer, metric_name, float_format='%.4f')

    src_path = src.config.path_config.test_results / 'applicability_2-11.xlsx'
    dst_path = src.config.path_config.test_results / 'applicability_final.xlsx'
    with pd.ExcelWriter(dst_path) as writer:
        for metric_name in ['F1', 'AUC', 'G-Mean']:
            df = pd.read_excel(src_path, index_col=0, sheet_name=metric_name).loc[DATASETS]
            df.to_excel(writer, metric_name)
            df.style.apply(highlight_higher_cells, axis=1).to_excel(writer, metric_name, float_format='%.4f')

    src_path = src.config.path_config.test_results / 'vstm_2-14.xlsx'
    dst_path = src.config.path_config.test_results / 'vstm_final.xlsx'
    with pd.ExcelWriter(dst_path) as writer:
        for metric_name in ['F1', 'AUC', 'G-Mean']:
            df = pd.read_excel(src_path, index_col=0, sheet_name=metric_name).loc[DATASETS]
            df.to_excel(writer, metric_name)
            df.style.highlight_max(axis=1).to_excel(writer, metric_name, float_format='%.4f')
