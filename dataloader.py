"""Load data to be used for training"""
import json
import h5py
import numpy as np
from collections import defaultdict


class DataLoader(object):
    def __init__(self, args, subsets):
        self.dialogue_fetch_counter = defaultdict(lambda: 0)

        self.data = {}

        # Load input json data, containing ind2word, word2ind, etc..
        for key, value in json.load(open(args.input_json, 'r')).items():
            self.data[key] = value

        # Add <start> and <end> to the vocabulary
        count = len(self.data['word2ind'])
        self.data['word2ind']['<START>'] = count
        self.data['word2ind']['<EOS>'] = count + 1
        self.vocab_size = count + 2

        print('Vocabulary size (with <start>, <end>: {}'.format(self.vocab_size))

        # Construct ind2word
        ind2word = {}
        for word, ind in self.data['word2ind'].items():
            ind2word[ind] = word
        self.data['ind2word'] = ind2word

        # Read questions, answers and options
        print('Data loader loading h5 file: ', args.input_data)
        data_file = h5py.File(args.input_data, 'r')
        imgFile = h5py.File(args.input_img, 'r')

        self.num_threads = {}
        for subset in subsets:
            # Read captions
            self.data[subset + '_cap'] = np.array(data_file['cap_' + subset])
            self.data[subset + '_cap_len'] = np.array(data_file['cap_length_' + subset])

            # Read question related information
            self.data[subset + '_ques'] = np.array(data_file['ques_' + subset])
            self.data[subset + '_ques_len'] = np.array(data_file['ques_length_' + subset])

            # Read answer related information
            self.data[subset + '_ans'] = np.array(data_file['ans_' + subset])
            self.data[subset + '_ans_len'] = np.array(data_file['ans_length_' + subset])
            self.data[subset + '_ans_ind'] = np.array(data_file['ans_index_' + subset])

            print('Reading image features...')
            img_feats = np.array(imgFile['/images_' + subset])

            # Normalize the image features if needed
            if args.img_norm == 1:
                print('Normalizing image features..')
                nm = np.sqrt(np.sum(np.multiply(img_feats, img_feats), 1))[:,
                     np.newaxis]
                img_feats /= nm.astype(np.float)

            self.data[subset + '_img_fv'] = img_feats.astype(np.float16)
            self.data[subset + '_img_pos'] = np.array(data_file['img_pos_' + subset])

            print('{}: No. of threads: {}; '
                  'No of rounds: {}, '
                  'Max ques len: {}, '
                  'Max ans len: {}'.format(subset,
                                           self.data[subset + '_ques'].shape[0],
                                           self.data[subset + '_ques'].shape[1],
                                           self.data[subset + '_ques'].shape[2],
                                           self.data[subset + '_ans'].shape[2]))

            if subset == 'train':
                self.num_train_threads = self.data['train_ques'].shape[0]
                self.num_threads['train'] = self.num_train_threads

            if subset == 'test':
                self.num_test_threads = self.data['test_ques'].shape[0]
                self.num_threads['test'] = self.num_test_threads

            if subset == 'val':
                self.num_val_threads = self.data['val_ques'].shape[0]
                self.num_threads['val'] = self.num_val_threads

            # Record the options
            if subset in ['train', 'val', 'test']:
                self.data[subset + '_opt'] = np.array(data_file['opt_' + subset])
                self.data[subset + '_opt_len'] = np.array(
                    data_file['opt_length_' + subset])
                self.data[subset + '_opt_list'] = np.array(
                    data_file['opt_list_' + subset])

            # Assume similar stats across multiple data subsets
            # Maximum number of question per image, ideally 10
            self.max_ques_count = self.data[subset + '_ques'].shape[1]

            # Maximum length of questions
            self.max_ques_len = self.data[subset + '_ques'].shape[2]

            # Maximum length of answer
            self.max_ans_len = self.data[subset + '_ans'].shape[1]

            # Number of options, if read
            if self.data[subset + '_opt'] is not None:
                self.num_options = self.data[subset + '_opt'].shape[2]

            self.data[subset + '_cap'] = np.array(data_file['cap_' + subset])
            self.data[subset + '_cap_len'] = np.array(data_file['cap_length_' + subset])

        data_file.close()
        imgFile.close()

        print('Data loaded')
        self.max_history_len = args.max_history_len or 60

        for subset in subsets:
            self.prepare_dataset(subset)

    def prepare_dataset(self, subset):
        """
        Prepare captions, questions and answers for retrieval.
        Questions and captions: Right align.
        Answers: Prefix with <start> and <end>
        """
        print('Processing dataset ', subset)
        # print('Right aligning questions: '.format(subset))
        # self.data[subset + '_ques_fwd'] = right_align(self.data[subset + '_ques'],
        #                                               self.data[subset + '_ques_len'])
        #
        # self.data[subset + '_cap_fwd'] = right_align(self.data[subset + '_cap'],
        #                                              self.data[subset + '_cap_len'])

        # If separate captions are needed
        self.process_history(subset)
        # self.process_answers(subset)
        # self.process_options(subset)
        self.process_full_answers(subset)
        self.process_full_questions(subset)
        self.process_full_history(subset)

    def process_full_history(self, subset):
        print('Processing full history of subset ', subset)
        captions = self.data[subset + '_cap']
        cap_len = self.data[subset + '_cap_len']

        questions = self.data[subset + '_ques']
        ques_len = self.data[subset + '_ques_len']

        answers = self.data[subset + '_ans']
        ans_len = self.data[subset + '_ans_len']

        num_convs = answers.shape[0]
        num_rounds = answers.shape[1]

        max_ques_len = questions.shape[2]
        max_ans_len = answers.shape[2]

        max_ques_len = max_ques_len + 1
        max_ans_len = max_ans_len + 1

        history = np.zeros(shape=(num_convs, num_rounds * (max_ques_len + max_ans_len)),
                           dtype=np.long)
        history_len = np.zeros(shape=(num_convs, 1), dtype=np.long)

        eos_symbol = self.data['word2ind']['<EOS>']
        for th_id in range(num_convs):
            len_c = cap_len[th_id]
            marker = 0
            for round_id in range(num_rounds):
                if round_id == 0:
                    history[th_id][marker:len_c] = captions[th_id][:len_c]
                    history[th_id][len_c] = eos_symbol
                    marker += (len_c + 1)
                else:
                    len_q = ques_len[th_id][round_id - 1]
                    len_a = ans_len[th_id][round_id - 1]

                    if len_q > 0:
                        history[th_id][marker: marker + len_q] = questions[th_id][
                                                                     round_id - 1][:len_q]
                        marker += len_q

                    if len_a > 0:
                        history[th_id][marker: marker + len_a] = answers[th_id][
                                                                     round_id - 1][:len_a]
                        history[th_id][marker + len_a] = eos_symbol
                        marker += (len_a + 1)

            history_len[th_id] = marker

        self.data[subset + '_history_full'] = history
        self.data[subset + '_history_full_len'] = history_len

    def process_full_questions(self, subset):
        print('Processing questions of subset ', subset)
        questions = self.data[subset + '_ques']
        ques_len = self.data[subset + '_ques_len']

        num_convs = questions.shape[0]
        num_rounds = questions.shape[1]
        max_ques_len = questions.shape[2]

        # All questions are now appended with an <EOS> token
        # therefore their length is increased by 1
        new_max_ques_len = max_ques_len + 1

        # Shape of questions dialog
        ques_shape = (num_convs, new_max_ques_len * num_rounds)
        ques_dialog = np.zeros(shape=ques_shape, dtype=np.long)
        ques_dialog_lengths = np.zeros(shape=(num_convs, 1), dtype=np.long)

        eos_symbol = self.data['word2ind']['<EOS>']
        for th_id in range(num_convs):
            marker = 0
            for round_id in range(num_rounds):
                ques_length = ques_len[th_id][round_id]
                if ques_length > 0:
                    ques_dialog[th_id][marker:marker + ques_length] = questions[th_id][
                                                                          round_id][
                                                                      :ques_length]
                    ques_dialog[th_id][marker + ques_length] = eos_symbol
                    marker += (ques_length + 1)
                else:
                    raise Exception
                ques_dialog_lengths[th_id] = marker

        self.data[subset + '_full_ques'] = ques_dialog
        self.data[subset + '_full_ques_len'] = ques_dialog_lengths

    def process_full_answers(self, subset):
        answers = self.data[subset + '_ans']
        ans_len = self.data[subset + '_ans_len']

        num_convs = answers.shape[0]
        num_rounds = answers.shape[1]
        max_ans_len = answers.shape[2]

        new_max_ans_len = max_ans_len + 1

        ans_shape = (num_convs, new_max_ans_len * num_rounds)
        ans_dialog = np.zeros(shape=ans_shape, dtype=np.long)
        ans_dialog_lengths = np.zeros(shape=(num_convs, 1), dtype=np.long)

        eos_symbol = self.data['word2ind']['<EOS>']
        for th_id in range(num_convs):
            marker = 0
            for round_id in range(num_rounds):
                ans_length = ans_len[th_id][round_id]
                if ans_length > 0:
                    ans_dialog[th_id][marker:marker + ans_length] = answers[th_id][
                                                                        round_id][
                                                                    :ans_length]
                    ans_dialog[th_id][marker + ans_length] = eos_symbol
                    marker += (ans_length + 1)
                else:
                    raise Exception
                ans_dialog_lengths[th_id] = marker

        self.data[subset + '_full_ans'] = ans_dialog
        self.data[subset + '_full_ans_len'] = ans_dialog_lengths

    def process_answers(self, subset):
        print('Processing answers of subset ', subset)
        # Prefix answers with <START>, <END>; adjust answer lengths
        answers = self.data[subset + '_ans']
        ans_len = self.data[subset + '_ans_len']

        num_convs = answers.shape[0]
        num_rounds = answers.shape[1]
        max_ans_len = answers.shape[2]

        decode_in = np.zeros((num_convs, num_rounds, max_ans_len + 1), dtype=np.long)
        decode_out = np.zeros((num_convs, num_rounds, max_ans_len + 1), dtype=np.long)

        # Decode_in begins with <start>
        decode_in[:, :, 0] = self.data['word2ind']['<START>']

        # Go over each answer and modify
        end_token_id = self.data['word2ind']['<END>']
        for th_id in range(num_convs):
            for round_id in range(num_rounds):
                length = ans_len[th_id][round_id]

                # Only if nonzero
                if length > 0:
                    decode_in[th_id][round_id][1:length + 1] = answers[th_id][round_id][
                                                               :length]
                    decode_out[th_id][round_id][:length] = answers[th_id][round_id][
                                                           :length]
                    decode_out[th_id][round_id][length] = end_token_id
                else:
                    print('Warning: empty answer at ({} {} {})'.format(th_id, round_id,
                                                                       length))

        self.data[subset + '_ans_len_normal'] = np.copy(self.data[subset + '_ans_len'])
        self.data[subset + '_ans_len'] += 1
        self.data[subset + '_ans_in'] = decode_in
        self.data[subset + '_ans_out'] = decode_out

    def process_history(self, subset):
        print('Processing history of subset ', subset)

        captions = self.data[subset + '_cap']
        questions = self.data[subset + '_ques']
        ques_len = self.data[subset + '_ques_len']
        cap_len = self.data[subset + '_cap_len']
        max_ques_len = questions.shape[2]

        answers = self.data[subset + '_ans']
        ans_len = self.data[subset + '_ans_len']
        num_convs = answers.shape[0]
        num_rounds = answers.shape[1]
        max_ans_len = answers.shape[2]

        history = np.zeros((num_convs, num_rounds, max_ques_len + max_ans_len),
                           dtype=np.long)
        history_len = np.zeros((num_convs, num_rounds), dtype=np.long)

        # Go over each questions and append it with answer
        for th_id in range(num_convs):
            len_c = cap_len[th_id]
            len_h = 0
            for round_id in range(num_rounds):
                if round_id == 0:
                    # First round has caption as history
                    history[th_id][round_id][:max_ques_len + max_ans_len] = \
                        captions[th_id][:max_ques_len + max_ans_len]
                    len_h = int(min(len_c, max_ques_len + max_ans_len))
                else:
                    len_q = ques_len[th_id][round_id - 1]
                    len_a = ans_len[th_id][round_id - 1]

                    if len_q > 0:
                        history[th_id][round_id][:len_q] = questions[th_id][
                                                               round_id - 1][:len_q]

                    if len_a > 0:
                        history[th_id][round_id][len_q: len_q + len_a] = \
                            answers[th_id][round_id - 1][:len_a]
                    len_h = len_a + len_q

                # Save the history length
                history_len[th_id][round_id] = len_h

        # Right align history and then save
        print('Right aligning history: ', subset)
        #	self.data[subset + '_hist'] = right_align(history, history_len)
        #	self.data[subset + '_hist_len'] = history_len

        # Also save normally-aligned history
        self.data[subset + '_hist_normal'] = history
        self.data[subset + '_hist_normal_len'] = history_len

    def process_options(self, subset):
        print('Processing options of subset ', subset)
        lengths = self.data[subset + '_opt_len']
        answers = self.data[subset + '_ans']
        max_ans_len = answers.shape[2]
        answers = self.data[subset + '_opt_list']
        num_convs = answers.shape[0]

        ans_list_len = answers.shape[0]
        decode_in = np.zeros((ans_list_len, max_ans_len + 1), dtype=np.long)
        decode_out = np.zeros((ans_list_len, max_ans_len + 1), dtype=np.long)

        # Decode in begins with <START>
        decode_in[:, 0] = self.data['word2ind']['<START>']

        # Go over each answer and modify
        end_token_id = self.data['word2ind']['<END>']
        for _id in range(ans_list_len):
            # Print progress for nubmer of images
            if _id % 10000 == 0:
                print('Progress {}/{}'.format(_id, num_convs))

            length = lengths[_id]

            # Only if nonzero
            if length > 0:
                decode_in[_id][1:length + 1] = answers[_id][:length]
                decode_out[_id][:length] = answers[_id][:length]
                decode_out[_id][length] = end_token_id
            else:
                print('Warning: empty answer for {} at {}'.format(subset, _id))

        self.data[subset + '_opt_len'] = self.data[subset + '_opt_len'] + 1
        self.data[subset + '_opt_in'] = decode_in
        self.data[subset + '_opt_out'] = decode_out

    def get_train_batch(self, batch_size, right_aligned=False, time_major=True):
        size = batch_size
        inds = np.random.choice(self.num_train_threads - 1, size)

        # keep track of how many times each dialogue has been fetched
        for idx in inds:
            self.dialogue_fetch_counter[idx] += 1

        batch_rounds = self.get_index_data(inds, 'train', right_aligned, time_major)
        batch_full = self.get_full_index_data(inds, 'train', time_major)
        return batch_rounds, batch_full

    def get_test_batch(self, start_id, batch_size, subset='val', right_aligned=False,
                       time_major=True, full_dialogue=False):

        # Get the next start id and fill up currrent indices till then
        if subset == 'val':
            next_start_id = min(self.num_val_threads + 1, start_id + batch_size)
        elif subset == 'test':
            next_start_id = min(self.num_test_threads + 1, start_id + batch_size)
        else:
            raise Exception

        inds = np.array([i for i in range(start_id, next_start_id)])

        # Index question, answers, image features for batch
        batch_rounds = self.get_index_data(inds, subset, right_aligned, time_major)
        batch_full = self.get_full_index_data(inds, subset, time_major)

        return batch_rounds, batch_full, next_start_id

    def get_full_index_data(self, inds, subset, time_major=True):
        question_lengths = np.take(self.data[subset + '_full_ques_len'], inds, axis=0)
        max_ques_length = int(np.max(question_lengths))
        questions = np.take(self.data[subset + '_full_ques'], inds, axis=0)[:,
                    0:max_ques_length]

        answer_lengths = np.take(self.data[subset + '_full_ans_len'], inds, axis=0)
        max_ans_length = int(np.max(answer_lengths))
        answers = np.take(self.data[subset + '_full_ans'], inds, axis=0)[:,
                  0:max_ans_length]

        history_lengths = np.take(self.data[subset + '_history_full_len'], inds,
                                  axis=0)
        max_hist_length = int(np.max(history_lengths))
        history = np.take(self.data[subset + '_history_full'], inds, axis=0)[:,
                  0:max_hist_length]

        img_inds = np.take(self.data[subset + '_img_pos'], inds, axis=0)
        img_feats = np.take(self.data[subset + '_img_fv'], img_inds, axis=0)

        if time_major:
            questions = np.swapaxes(questions, 0, 1)
            answers = np.swapaxes(answers, 0, 1)
            history = np.swapaxes(history, 0, 1)

        output = {
            'question': questions,
            'question_lengths': question_lengths,
            'answer': answers,
            'answer_lengths': answer_lengths,
            'history': history,
            'history_lengths': history_lengths,
            'img_inds': img_inds,
            'img_feats': img_feats
        }

        return output

    def get_index_data(self, inds, subset, right_aligned=False, time_major=False):
        if not right_aligned:
            question_lengths = np.take(self.data[subset + '_ques_len'], inds, axis=0)
            max_ques_length = int(np.max(question_lengths))
            question = np.take(self.data[subset + '_ques'], inds, axis=0)[:, :,
                       0:max_ques_length]

            answer_lengths = np.take(self.data[subset + '_ans_len'], inds, axis=0)
            max_answ_length = int(np.max(answer_lengths))
            answer = np.take(self.data[subset + '_ans'], inds, axis=0)[:, :,
                     0:max_answ_length]

            history_lengths = np.take(self.data[subset + '_hist_normal_len'], inds,
                                      axis=0)
            max_hist_length = int(np.max(history_lengths))
            history = np.take(self.data[subset + '_hist_normal'], inds, axis=0)[:, :,
                      0:max_hist_length]

            img_inds = np.take(self.data[subset + '_img_pos'], inds, axis=0)
            img_feats = np.take(self.data[subset + '_img_fv'], img_inds, axis=0)

            if time_major:
                question = np.swapaxes(question, 0, 2)
                answer = np.swapaxes(answer, 0, 2)
                history = np.swapaxes(history, 0, 2)

            output = {
                'question': question,
                'question_lengths': question_lengths,
                'answer': answer,
                'answer_lengths': answer_lengths,
                'history': history,
                'history_lengths': history_lengths
            }

            output['img_inds'] = img_inds
            output['img_feats'] = img_feats

            return output
        else:
            # Get the question lengths
            batch_ques_len = np.take(self.data[subset + '_ques_len'], inds, axis=0)
            max_ques_len = int(np.max(batch_ques_len))

            # Get questions
            ques_fwd = np.take(self.data[subset + '_ques_fwd'], inds, axis=0)
            # ques_fwd = ques_fwd[:, :, -max_ques_len:]

            history = None
            batch_hist_len = np.take(self.data[subset + '_hist_len'], inds, axis=0)
            max_hist_len = int(min(np.max(batch_hist_len), self.max_history_len))
            history = np.take(self.data[subset + '_hist'], inds, axis=0)
            # history = history[:, :,-max_hist_len:]

            img_feats = None
            img_inds = np.take(self.data[subset + '_img_pos'], inds, axis=0)
            img_feats = np.take(self.data[subset + '_img_fv'], img_inds, axis=0)

            # Get the answer lengths
            batch_ans_len = np.take(self.data[subset + '_ans_len'], inds, axis=0)
            max_ans_len = int(np.max(batch_ans_len))

            # Answer labels (decode input and output)
            answer_in = np.take(self.data[subset + '_ans_in'], inds, axis=0)
            # answer_in = answer_in[:, :, :max_ans_len]
            answer_out = np.take(self.data[subset + '_ans_out'], inds, axis=0)
            # answer_out = answer_out[:, :, :max_ans_len]
            answer_ind = np.take(self.data[subset + '_ans_ind'], inds, axis=0)

            output = {}
            output['ques_fwd'] = ques_fwd
            output['answer_in'] = answer_in
            output['answer_out'] = answer_out
            output['answer_ind'] = answer_ind

            if history is not None:
                output['hist'] = history

            if img_feats is not None:
                output['img_feat'] = img_feats
                output['img_inds'] = img_inds

            return output

    def get_index_option(self, inds, subset):
        output = {}
        opt_inds = np.take(self.data[subset + '_opt'], inds, axis=0)
        ind_vector = opt_inds.reshape(-1)

        batch_opt_len = np.take(self.data[subset + '_opt_len'], ind_vector, axis=0)
        max_opt_len = int(np.max(batch_opt_len))

        option_in = np.take(self.data[subset + '_opt_in'], ind_vector, axis=0)
        option_in = option_in.reshape(
            (option_in.shape[0], option_in.shape[1], option_in.shape[2], -1))
        option_in = option_in[:, :, :, :max_opt_len]

        option_out = np.take(self.data[subset + '_opt_out'], ind_vector, axis=0)
        option_out = option_out.reshape(
            (option_out.shape[0], option_out.shape[1], option_out.shape[2], -1))
        option_out = option_out[:, :, :, :max_opt_len]

        output['option_in'] = option_in
        output['option_out'] = option_out

        return output
