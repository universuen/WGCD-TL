import os

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import src

TEST_NAME = 'test_script'

K = 5

METRICS = [
    'F1',
    'AUC',
    'G-Mean',
]

DATASETS = [
    'ecoli-0-1-4-7_vs_5-6.dat',
    'ecoli-0-1_vs_5.dat',
    'ecoli-0-6-7_vs_3-5.dat',
]

def highlight_legal_cells(s: pd.Series) -> list[str]:
    result_ = []
    if s[1] > s[0]:
        result_.extend(['', 'background-color: yellow'])
    else:
        result_.extend(['', ''])
    for i in s[2:-1]:
        if i > s[1]:
            result_.append('background-color: yellow')
        else:
            result_.append('')
    if s[-1] > max(s):
        result_.append('background-color: yellow')
    else:
        result_.append('')
    return result_


class SNGANSM(src.gans.SNGAN):
    def __init__(self):
        super().__init__()

    def _fit(self):
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=src.config.gan.d_lr,
            betas=(0.5, 0.999),
        )
        g_optimizer = torch.optim.Adam(
            params=self.g.parameters(),
            lr=src.config.gan.g_lr,
            betas=(0.5, 0.999),
        )

        x = src.datasets.PositiveDataset()[:][0].to(src.config.device)
        for _ in range(src.config.gan.epochs):
            for __ in range(src.config.gan.d_loops):
                self.d.zero_grad()
                prediction_real = self.d(x)
                loss_real = -torch.log(prediction_real.mean())
                z = torch.randn(len(x), src.models.z_size, device=src.config.device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = -torch.log(1 - prediction_fake.mean())
                loss = loss_real + loss_fake
                loss.backward()
                d_optimizer.step()
            for __ in range(src.config.gan.g_loops):
                self.g.zero_grad()
                real_x_hidden_output = self.d.hidden_output.detach()
                z = torch.randn(len(x), src.models.z_size, device=src.config.device)
                fake_x = self.g(z)
                final_output = self.d(fake_x)
                fake_x_hidden_output = self.d.hidden_output
                real_x_hidden_distribution = src.utils.normalize(real_x_hidden_output)
                fake_x_hidden_distribution = src.utils.normalize(fake_x_hidden_output)
                hidden_loss = torch.norm(
                    real_x_hidden_distribution - fake_x_hidden_distribution,
                    p=2
                ) * src.config.gan.hl_lambda
                loss = -torch.log(final_output.mean()) + hidden_loss
                loss.backward()
                g_optimizer.step()


if __name__ == '__main__':
    src.config.logger.level = 'WARNING'
    result_file = src.config.path.test_results / f'ablation_{TEST_NAME}.xlsx'
    if os.path.exists(result_file):
        input(f'{result_file} already existed, continue?')
    methods = [
        'Original',
        'SNGAN',
        'SNGAN-V',
        'SNGAN-SM',
        'SNGAN-R',
        'SNGAN-TL',
        'RVGAN-TL',
    ]
    # [metric][method][dataset]
    result = {
        k: pd.DataFrame(
            {
                kk: {
                    kkk: 0.0 for kkk in [*DATASETS, 'mean']
                } for kk in methods
            }
        ) for k in METRICS
    }
    for dataset_name in tqdm(DATASETS):
        # prepare data
        src.utils.set_random_state()
        samples, labels = src.utils.preprocess_data(dataset_name)
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=src.config.seed)
        temp_result = {
            k: {
                kk: [] for kk in methods
            } for k in METRICS
        }
        # k-fold test
        for training_indices, test_indices in skf.split(samples, labels):
            src.datasets.training_samples = samples[training_indices]
            src.datasets.training_labels = labels[training_indices]
            src.datasets.test_samples = samples[test_indices]
            src.datasets.test_labels = labels[test_indices]

            training_dataset = src.datasets.FullDataset(training=True)
            test_dataset = src.datasets.FullDataset(training=False)
            gan = src.gans.SNGAN()
            gan.fit()
            classifier: src.classifier.Classifier = None

            for method_name in methods:
                src.utils.set_random_state()
                if method_name == 'Original':
                    classifier = src.classifier.Classifier('Original')
                    classifier.fit(training_dataset)
                elif method_name == 'SNGAN':
                    gan_dataset = src.utils.get_gan_dataset(gan)
                    classifier = src.classifier.Classifier('SNGAN')
                    classifier.fit(gan_dataset)
                elif method_name == 'SNGAN-V':
                    vae = src.vae.VAE()
                    vae.fit()
                    pos_dataset = src.datasets.PositiveDataset().to(src.config.device)
                    neg_dataset = src.datasets.NegativeDataset().to(src.config.device)
                    target_dataset = src.datasets.FullDataset().to(src.config.device)
                    total_pos_cnt = len(pos_dataset)
                    total_neg_cnt = len(neg_dataset)
                    target_sample_num = total_neg_cnt - total_pos_cnt
                    if target_sample_num > 0:
                        z = vae.generate_z(target_sample_num)
                        new_samples = gan.generate_samples(z)
                        new_labels = torch.ones(target_sample_num, device=src.config.device)
                        target_dataset.samples = torch.cat(
                            [
                                target_dataset.samples,
                                new_samples,
                            ],
                        )
                        target_dataset.labels = torch.cat(
                            [
                                target_dataset.labels,
                                new_labels,
                            ]
                        )
                        target_dataset.samples = target_dataset.samples.detach()
                        target_dataset.labels = target_dataset.labels.detach()
                    classifier = src.classifier.Classifier('SNGAN-V')
                    classifier.fit(target_dataset)
                elif method_name == 'SNGAN-SM':
                    gan_sm = SNGANSM()
                    gan_sm.fit()
                    gan_sm_dataset = src.utils.get_gan_dataset(gan_sm)
                    classifier = src.classifier.Classifier('SNGAN-SM')
                    classifier.fit(gan_sm_dataset)
                elif method_name == 'SNGAN-R':
                    former_hl_lambda = src.config.gan.hl_lambda
                    src.config.gan.hl_lambda = 0
                    rgan = src.gans.RSNGAN()
                    rgan.fit()
                    src.config.gan.hl_lambda = former_hl_lambda
                    rgan_dataset = src.utils.get_gan_dataset(rgan)
                    classifier = src.classifier.Classifier('SNGAN-R')
                    classifier.fit(rgan_dataset)
                elif method_name == 'SNGAN-TL':
                    gan_dataset = src.utils.get_gan_dataset(gan)
                    classifier = src.tr_ada_boost.TrAdaBoost()
                    classifier.fit(gan_dataset, training_dataset)
                elif method_name == 'RVGAN-TL':
                    rgan = src.gans.RSNGAN()
                    rgan.fit()
                    rgan_tl_dataset = src.utils.get_rgan_dataset(rgan)
                    classifier = src.tr_ada_boost.TrAdaBoost()
                    classifier.fit(rgan_tl_dataset, training_dataset)

                classifier.test(test_dataset)
                for metric_name in METRICS:
                    temp_result[metric_name][method_name].append(classifier.metrics[metric_name])

            # calculate final metrics
            for method_name in methods:
                for metric_name in METRICS:
                    result[metric_name][method_name][dataset_name] = np.mean(temp_result[metric_name][method_name])
            # calculate average metrics on all datasets
            for method_name in methods:
                for metric_name in METRICS:
                    result[metric_name][method_name]['mean'] = np.mean(
                        [i for i in result[metric_name][method_name].values])
            # write down current result
            with pd.ExcelWriter(result_file) as writer:
                for metric_name in METRICS:
                    df = result[metric_name]
                    df.to_excel(writer, metric_name)
                    df.style.apply(highlight_legal_cells, axis=1).to_excel(writer, metric_name, float_format='%.4f')
