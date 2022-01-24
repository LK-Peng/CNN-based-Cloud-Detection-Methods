import os
import json
import pandas as pd
from collections import OrderedDict


class Tracker(object):

    def __init__(self, run_directory, run_filename='training_loop_performance'):
        self.out_csv_file = os.path.join(run_directory, run_filename + '.csv')
        self.out_json_file = os.path.join(run_directory, run_filename + '.json')
        self.run_data = []
        self.results_epoch = OrderedDict()

    def begin_epoch(self):
        self.results_epoch = OrderedDict()

    def train_epoch(self, epoch, train_loss, lr):
        self.results_epoch['epoch'] = epoch
        self.results_epoch['train loss'] = train_loss
        self.results_epoch['learning rate'] = lr

    def val_epoch(self, epoch, val_loss, pa, mpa, miou, fwiou):
        assert epoch == self.results_epoch['epoch']
        self.results_epoch['val loss'] = val_loss
        self.results_epoch['PA'] = pa
        self.results_epoch['MPA'] = mpa
        self.results_epoch['MIoU'] = miou
        self.results_epoch['FWIoU'] = fwiou

    def end_epoch(self):
        self.run_data.append(self.results_epoch)
        self.results_epoch = OrderedDict()
        self._save()

    def _save(self):
        # save as csv file
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(self.out_csv_file)

        # save as json file
        with open(self.out_json_file, 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
