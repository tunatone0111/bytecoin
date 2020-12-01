import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf

import pandas as pd
import numpy as np
import random
import time
import datetime
import os
import argparse
import torch.nn.functional as F

#train = pd.read_csv('/home/awefjio12345/Downloads/nsmc-master/ratings_train.txt',sep = '\t')
#test = pd.read_csv('/home/awefjio12345/Downloads/nsmc-master/ratings_test.txt',sep = '\t')
#test = pd.read_csv('/home/awefjio12345/Downloads/nsmc-master/test.csv',sep = ',',names=['document','label'],header=None)

train_set_location = 'your train set file location'
test_set_location = 'your test set file location'
model_save_location = 'your model save location'
model_load_location = 'your model load location'

train = pd.read_csv(train_set_location,sep = '\t')
test = pd.read_csv(test_set_location,sep = '\t')

class Bert_classification():
    def __init__(self,epoch,batch_size):
        # 입력 토큰의 최대 시퀀스 길이
        self.MAX_LEN = 128
        self.batch_size = batch_size

        # GPU 디바이스 이름 구함
        self.device_name = tf.test.gpu_device_name()

        # GPU 디바이스 이름 검사
        if self.device_name == '/device:GPU:0':
            print('Found GPU at: {}'.format(self.device_name))
        else:
            raise SystemError('GPU device not found')

        # 디바이스 설정
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print('No GPU available, using the CPU instead.')

        # 분류를 위한 BERT 모델 생성
        self.model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
        #self.model.cuda()

        checkpoint = torch.load(model_load_location)
        self.model.load_state_dict(checkpoint['model'])
        self.model.cuda()

        # 옵티마이저 설정
        self.optimizer = AdamW(self.model.parameters(),
                          lr=2e-5,  # 학습률
                          eps=1e-8  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                          )

        # 에폭수
        self.epochs = epoch

        #훈련중프린트주기
        self.train_print_period = 500

        #테스트중프린트주기
        self.test_print_period = 100

    def sentences_conversion(self,raw_texts):
        # 리뷰 문장 추출
        sentences = raw_texts['document']
        print('setenses is :   ',sentences)
        print(type(sentences))

        # BERT의 입력 형식에 맞게 변환
        sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
        sentences[:10]

        # 라벨 추출
        labels = raw_texts['label'].values
        print('labels is :    ',labels)

        # BERT의 토크나이저로 문장을 토큰으로 분리
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        print(sentences[0])
        print(tokenized_texts[0])

        # 토큰을 숫자 인덱스로 변환
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

        # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
        input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN, dtype="long", truncating="post",
                                        padding="post")
        print(input_ids[0])

        # 어텐션 마스크 초기화
        attention_masks = []

        # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
        # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        print(attention_masks[0])

        return input_ids,attention_masks,labels

    def get_ready_4realwork(self,contents_lst):


        # BERT의 입력 형식에 맞게 변환
        sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in contents_lst]
        sentences[:10]
        # BERT의 토크나이저로 문장을 토큰으로 분리
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        print(sentences[0])
        print(tokenized_texts[0])

        # 토큰을 숫자 인덱스로 변환
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

        # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
        input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN, dtype="long", truncating="post",
                                  padding="post")
        print(input_ids[0])

        # 어텐션 마스크 초기화
        attention_masks = []

        # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
        # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        print(attention_masks[0])

        # 데이터를 파이토치의 텐서로 변환
        work_inputs = torch.tensor(input_ids)
        work_masks = torch.tensor(attention_masks)

        print(work_inputs[0])
        print(work_masks[0])

        # 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
        # 학습시 배치 사이즈 만큼 데이터를 가져옴
        work_data = TensorDataset(work_inputs, work_masks)
        work_sampler = SequentialSampler(work_data)
        work_dataloader = DataLoader(work_data, sampler=work_sampler, batch_size=self.batch_size)

        return work_data, work_sampler, work_dataloader

    def get_train_validation_set(self,raw_texts):
        input_ids, attention_masks, labels = self.sentences_conversion(raw_texts=raw_texts)

        # 훈련셋과 검증셋으로 분리
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                            labels,
                                                                                            random_state=2018,
                                                                                            test_size=0.1)

        # 어텐션 마스크를 훈련셋과 검증셋으로 분리
        train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                               input_ids,
                                                               random_state=2018,
                                                               test_size=0.1)

        print('train_input is : ',train_inputs.shape)
        print('train_labesl is : ',train_labels.shape)
        print('train_masks is : ',len(train_masks))
        # 데이터를 파이토치의 텐서로 변환
        train_inputs = torch.tensor(train_inputs)
        train_labels = torch.tensor(train_labels)
        train_masks = torch.tensor(train_masks)
        validation_inputs = torch.tensor(validation_inputs)
        validation_labels = torch.tensor(validation_labels)
        validation_masks = torch.tensor(validation_masks)

        print(train_inputs[0])
        print(train_labels[0])
        print(train_masks[0])
        print(validation_inputs[0])
        print(validation_labels[0])
        print(validation_masks[0])

        # 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
        # 학습시 배치 사이즈 만큼 데이터를 가져옴
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)

        return train_data,train_sampler,train_dataloader,validation_data,validation_sampler,validation_dataloader

    def get_test_set(self,raw_texts):
        input_ids, attention_masks, labels = self.sentences_conversion(raw_texts=raw_texts)

        print('test_inputs : ',input_ids.shape)
        print('test_labels : ',labels.shape)
        print('test_masks : ',len(attention_masks))

        # 데이터를 파이토치의 텐서로 변환
        test_inputs = torch.tensor(input_ids)
        test_labels = torch.tensor(labels)
        test_masks = torch.tensor(attention_masks)

        print(test_inputs[0])
        print(test_labels[0])
        print(test_masks[0])

        # 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
        # 학습시 배치 사이즈 만큼 데이터를 가져옴
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)

        return test_data,test_sampler,test_dataloader

    # 정확도 계산 함수
    def flat_accuracy(self,preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    # 시간 표시 함수
    def format_time(self,elapsed):
        # 반올림
        elapsed_rounded = int(round((elapsed)))

        # hh:mm:ss으로 형태 변경
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train_and_validate(self):
        train_data, train_sampler, train_dataloader, validation_data, validation_sampler, validation_dataloader\
            = self.get_train_validation_set(train)

        # 총 훈련 스텝 : 배치반복 횟수 * 에폭
        self.total_steps = len(train_dataloader) * self.epochs

        # 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=self.total_steps)

        # 재현을 위해 랜덤시드 고정
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # 그래디언트 초기화
        self.model.zero_grad()

        # 에폭만큼 반복
        for epoch_i in range(0, self.epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # 시작 시간 설정
            t0 = time.time()

            # 로스 초기화
            total_loss = 0

            # 훈련모드로 변경
            self.model.train()

            # 데이터로더에서 배치만큼 반복하여 가져옴
            for step, batch in enumerate(train_dataloader):
                print('Training step : ',step,'.................','of epoch : ',epoch_i)
                self.model.train()

                two_for_break = False

                # 경과 정보 표시
                if step % self.train_print_period == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                    # ========================================
                    #               Validation
                    # ========================================
                    print("")
                    print("Running Validation...")

                    # 시작 시간 설정
                    t0 = time.time()

                    # 평가모드로 변경
                    self.model.eval()

                    # 변수 초기화
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0

                    # 데이터로더에서 배치만큼 반복하여 가져옴
                    for batch in validation_dataloader:
                        # 배치를 GPU에 넣음
                        batch = tuple(t.to(self.device) for t in batch)

                        # 배치에서 데이터 추출
                        b_input_ids, b_input_mask, b_labels = batch

                        ###############Bug fix code_if you run this code on colab or ubuntu, you don't need this####################
                        #b_input_ids = b_input_ids.type(torch.LongTensor)
                        #b_input_mask = b_input_mask.type(torch.LongTensor)
                        #b_labels = b_labels.type(torch.LongTensor)

                        #b_input_ids = b_input_ids.to(self.device)
                        #b_input_mask = b_input_mask.to(self.device)
                        #b_labels = b_labels.to(self.device)
                        ############################################

                        # 그래디언트 계산 안함
                        with torch.no_grad():
                            # Forward 수행
                            outputs = self.model(b_input_ids,
                                                 token_type_ids=None,
                                                 attention_mask=b_input_mask)

                        # 로스 구함
                        logits = outputs[0]

                        # CPU로 데이터 이동
                        logits = logits.detach().cpu().numpy()
                        label_ids = b_labels.to('cpu').numpy()

                        # 출력 로짓과 라벨을 비교하여 정확도 계산
                        tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                        eval_accuracy += tmp_eval_accuracy
                        nb_eval_steps += 1

                    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
                    if eval_accuracy/nb_eval_steps >=0.86:
                        print('saving models.....')
                        two_for_break = True
                        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
                                   model_save_location +str(step)+'_' +str(epoch_i + 1)+str(eval_accuracy/nb_eval_steps)+'2020_11_23' + '.tar')
                        print('saving models completed!')
                    print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

                if two_for_break ==True:
                    print('breaking out successed !')
                    break

                # 배치를 GPU에 넣음
                batch = tuple(t.to(self.device) for t in batch)
                self.model.train()

                # 배치에서 데이터 추출
                b_input_ids, b_input_mask, b_labels = batch

                ###############Bug fix code_when you run this code on colab or ubuntu, you don't need this####################
                #b_input_ids = b_input_ids.type(torch.LongTensor)
                #b_input_mask = b_input_mask.type(torch.LongTensor)
                #b_labels = b_labels.type(torch.LongTensor)

                #b_input_ids = b_input_ids.to(self.device)
                #b_input_mask = b_input_mask.to(self.device)
                #b_labels = b_labels.to(self.device)
                ##############################################################################################################

                # Forward 수행
                outputs = self.model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

                # 로스 구함
                loss = outputs[0]

                # 총 로스 계산
                total_loss += loss.item()

                # Backward 수행으로 그래디언트 계산
                loss.backward()

                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 그래디언트를 통해 가중치 파라미터 업데이트
                self.optimizer.step()

                # 스케줄러로 학습률 감소
                self.scheduler.step()

                # 그래디언트 초기화
                self.model.zero_grad()

            # 평균 로스 계산
            avg_train_loss = total_loss / len(train_dataloader)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(self.format_time(time.time() - t0)))



        print("")
        print("Training complete!")

    def test(self):
        test_data, test_sampler, test_dataloader = self.get_test_set(test)

        # 시작 시간 설정
        t0 = time.time()

        # 평가모드로 변경
        self.model.eval()

        # 변수 초기화
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # 데이터로더에서 배치만큼 반복하여 가져옴
        for step, batch in enumerate(test_dataloader):
            # 경과 정보 표시
            if step % self.test_print_period == 0 and not step == 0:
                elapsed = self.format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

            # 배치를 GPU에 넣음
            batch = tuple(t.to(self.device) for t in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch

            # 그래디언트 계산 안함
            with torch.no_grad():
                # Forward 수행
                outputs = self.model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # 로스 구함
            logits = outputs[0]

            # CPU로 데이터 이동
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # 출력 로짓과 라벨을 비교하여 정확도 계산
            tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("")
        print("Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("Test took: {:}".format(self.format_time( ( (time.time() - t0) ) ) ) )

    def work(self,contents_lst):
        work_data, work_sampler, work_dataloader = self.get_ready_4realwork(contents_lst=contents_lst)

        # 시작 시간 설정
        t0 = time.time()

        # 평가모드로 변경
        self.model.eval()

        # 변수 초기화
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        #lst to append labels
        label_lst = []

        # 데이터로더에서 배치만큼 반복하여 가져옴
        for step, batch in enumerate(work_dataloader):
            # 경과 정보 표시
            if step % self.test_print_period == 0 and not step == 0:
                elapsed = self.format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(work_dataloader), elapsed))

            # 배치를 GPU에 넣음
            batch = tuple(t.to(self.device) for t in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask = batch

            # 그래디언트 계산 안함
            with torch.no_grad():
                # Forward 수행
                outputs = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=b_input_mask)
            # 로스 구함
            logits = outputs[0]

            # CPU로 데이터 이동
            logits = logits.detach().cpu()

            softmaxed_logis = (F.softmax(logits,dim=1)).numpy()[:,1]

            print(f'length of logits is : {len(logits)}')

            label_lst += softmaxed_logis.tolist()

            nb_eval_steps += 1

        print("getting label for work took: {:}".format(self.format_time(((time.time() - t0)))))

        return label_lst

    def test_print(self):
        print('HI~')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, help='epoch number!')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, help='batch_size number!')
    parser.add_argument('-w', '--way', dest='way', default='train', type=str, choices=['train','test'],
                        help='what do you want to do? train or test?')
    args = parser.parse_args()

    epoch = args.epoch
    batch_size = args.batch_size
    way = args.way
    print(f'doing {way} with epcoh : {epoch} and batch_size : {batch_size}')

    bert = Bert_classification(epoch=epoch,batch_size=batch_size)

    if way == 'train':
        bert.train_and_validate()
    else:
        bert.test()












