# Copyright 2020 Seong Moon Jeong in IRIS lab. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from datetime import datetime
from time import time
import os

class LogManager:
    """
    Helper class for stdout and logging.
    Using this, you may see keras-like stdout.

    * Required arguments.
        - total number of epoch.
        - train loss, train accuracy
        - validation loss, validation accuracy

    * Optional arguments.
        - Total number of batches
          (if you don't give this information, LogManger automatically count batch number on first
           epoch and apply it to following epochs)
        - log file path
          (if it is not given, logging on a file doesn't occur)
    
    *** Usage ***
    from log_helper impor LogManager
    
    total_num_epoch = 100
    total_num_batch = total_num_train_data // batch_size
                      (+1, if you have a last batch which have less number of elements.)

    log_manager = LogManager(total_num_epoch, log_file_path='results/my_result.txt') 
    #           = LogManager(total_num_epoch, total_num_batch, 'results/my_results.txt')
    
    for epoch in range(1, total_num_epoch+1):
        total_loss = 0.0
        train_loss = 0.0
        started = False
    
        # Train one epoch.
        for data in train_dataset:
            if not started:     # Excute only once in a loop
                log_manager.print_epoch()
                started = True
    
            # Train one step(one batch). Get train_loss and train_accuracy.
            train_loss = ...
            train_accuracy = ...
    
            log_manaer.print_train_result(train_loss, train_accuracy) 
    
        # Validate. Get valid_loss and valid_accuracy.
        valid_loss = ...
        valid_accuracy = ...
    
        log_manager.print_comprehensive(valid_loss, valid_accuracy)

    """

    def __init__(self, total_num_epoch, total_num_batch=None, log_file_path=None):
        self.start_time = None
        self.one_epoch_elapse_time = None
        self.num_epoch = 0
        self.total_num_epoch = total_num_epoch
        self.total_num_batch = total_num_batch
        self.log_file_path = log_file_path

        # \033[A : move cursor up one line.
        # \r : move cursor to the beginning of the current line.
        # \033[F : move cursor to the beginning of the previous line.
        # \033[K : erase to end of line.
        self.train_progress_elapse_template = "\r\033[K" \
                            "{:7d}/{} - {} {:.3f}s/step"
        self.train_progress_eta_template = "\r\033[K" \
                            "{:7d}/{:d} - ETA {} {:.3f}s/step"
        self.comprehensive_template = self.train_progress_elapse_template \
                                    + " - {}: {:.2%}"
        self.log_template = "Epoch {:4d} - {}" \
                            " - {}: {:.2%}\n"


    # This method must excute once during one epoch.
    def print_epoch(self):
        self.num_epoch += 1

        self.start_time = time()
        self.current_batch = 0

        print("Epoch {:d}/{:d}".format(self.num_epoch, self.total_num_epoch))
    

    def print_train_result(self):
        self.current_batch += 1
        if self.start_time:
            self.elapse_time = time() - self.start_time
        else:
            raise Exception("[ERROR] Use print_epoch() first")

        self.period = self.elapse_time / self.current_batch

        if self.total_num_batch:
            self.elapse_timestamp = datetime.fromtimestamp(self.elapse_time).strftime("%H:%M:%S")
            eta = datetime.fromtimestamp((self.total_num_batch - self.current_batch)*self.period) \
                          .strftime("%H:%M:%S")
            result = self.train_progress_eta_template.format(self.current_batch,
                                                             self.total_num_batch,
                                                             eta, self.period)
        else:
            self.elapse_timestamp = datetime.fromtimestamp(self.elapse_time).strftime("%H:%M:%S")
            result = self.train_progress_elapse_template.format(self.current_batch,
                                                                'Unknown',
                                                                self.elapse_timestamp,
                                                                self.period)
        print(result, end='')

    
    def print_comprehensive(self, val_acc, train_mode=False):
        self.total_num_batch = self.current_batch
        self.valid_accuracy = val_acc
        self.one_epoch_elapse_time = self.elapse_time
        if train_mode is False:
            acc = 'test_acc'
        else:
            acc = 'train_acc'

        result = self.comprehensive_template.format(self.current_batch, self.total_num_batch,
                                                    self.elapse_timestamp, self.period, acc,
                                                    self.valid_accuracy)
        print(result)

        
        # Log on a file if log file path is given.
        if self.log_file_path:
            log = self.log_template.format(self.num_epoch, self.elapse_timestamp,
                                           acc, self.valid_accuracy)

            with open(self.log_file_path, 'a') as f:
                f.write(log)

    def print_configuration(self, FLAGS, ensemble=True):
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        
        config1 = '+++++++++++++++++++++Configuration++++++++++++++++++++++'
        config2 = 'DATASET: '+str(FLAGS.dataset)
        config3 = 'MODEL: '+str(FLAGS.model)
        config4 = 'BATCH SIZE: '+str(FLAGS.batch_size)+' at each gpu'
        config5 = 'QUANTIZATION MODE: '+str(FLAGS.q_mode)
        if ensemble:
            config6 = 'NORMALIZED FACTOR: '+str(1/FLAGS.factor)
            config7 = 'ITERATION: '+str(FLAGS.iteration)
            config8 = 'STOP POINT: '+str(FLAGS.stop_point)
            config9 = 'Training '+str(FLAGS.num_bit)+'&'+str(FLAGS.num_bit+FLAGS.up_bit)+' bit model.'
        else: 
            config6 = 'DROPOUT: '+str(FLAGS.dropout)
            config7 = 'LEARNING RATE: '+str(FLAGS.learning_rate)
            config8 = 'EPOCHS: '+str(FLAGS.epochs)
            config9 = 'Training '+str(FLAGS.num_bit)+' bit model.'
        config10 = '++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        config11 = ''
        for log in [config1, config2, config3, config4, config5, config6, config7, config8, config9, config10 ,config11]:
            print(log)
            with open(self.log_file_path, 'a') as f:
                f.write(log)
                f.write('\n')

    def _print(self, _string):
        print(_string)
        with open(self.log_file_path, 'a') as f:
            f.write(_string) 
            f.write('\n')

