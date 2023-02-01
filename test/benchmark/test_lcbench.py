import os
import unittest

from benchmarks.lcbench import LCBench


class TestLCBench(unittest.TestCase):

    def setUp(self) -> None:

        project_folder = os.path.expanduser(
            os.path.join(
                '~',
                'Desktop',
                'PhD',
                'Projekte',
                'DeepRegret',
            )
        )

        benchmark_data_path = os.path.join(
            project_folder,
            'lc_bench',
            'results',
            'data_2k.json',
        )

        self.lcbench = LCBench(benchmark_data_path)
        self.dataset_name = 'credit-g'

    def test_load_dataset_names(self):

        dataset_names = [
            'APSFailure', 'Amazon_employee_access', 'Australian',
            'Fashion-MNIST', 'KDDCup09_appetency', 'MiniBooNE',
            'adult', 'airlines', 'albert', 'bank-marketing',
            'blood-transfusion-service-center', 'car', 'christine',
            'cnae-9', 'connect-4', 'covertype', 'credit-g', 'dionis',
            'fabert', 'helena', 'higgs', 'jannis', 'jasmine',
            'jungle_chess_2pcs_raw_endgame_complete', 'kc1', 'kr-vs-kp',
            'mfeat-factors', 'nomao', 'numerai28.6', 'phoneme', 'segment',
            'shuttle', 'sylvine', 'vehicle', 'volkert',
        ]

        self.assertEqual(dataset_names, self.lcbench.dataset_names)

    def test_get_hyperparameter_candidates(self):

        hp_configs = self.lcbench.get_hyperparameter_candidates(self.dataset_name)
        self.assertEqual(hp_configs.shape, (LCBench.nr_hyperparameters, len(LCBench.param_space)))

    def test_get_performance(self):

        hp_index = 0
        self.assertGreaterEqual(
            self.lcbench.get_performance(self.dataset_name, hp_index, LCBench.max_budget),
            self.lcbench.get_performance(self.dataset_name, hp_index, 1),
        )
