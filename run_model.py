import argparse
import math
import os
import time

import dill
import logger
import numpy as np
import scipy.stats as stats
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell import LSTMCell, MultiRNNCell
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from tensorflow.python.util import nest

from round_based import dataloader


def ranks(predictions, dataset, true_inds, sqrt=True):
	"""
	:param predictions: [batch_size, image_feats]
	:param dataset: [dataset_size, image_feats]
	:param true_inds: [batch_size, 1]
	:param sqrt: Euclidian distance if True, otherwise Squared Euclidian Distance
	:return: Ranks
	"""
	d = (predictions ** 2).sum(axis=-1)[:, np.newaxis] + (dataset ** 2).sum(axis=-1)
	d -= 2 * np.squeeze(predictions.dot(dataset[..., np.newaxis]), axis=-1)

	if sqrt:
		d **= 0.5

	sorted_norms = np.argsort(d, axis=-1).astype(np.uint32)
	ranks = np.where(sorted_norms == true_inds[:, np.newaxis])[1]
	# reciprocal_ranks = 1. / ranks
	return ranks.tolist()


class ABOT(object):
	def __init__(self,
	             session,
	             config,
	             mode):
		assert mode.lower() in ['train', 'decode', 'rank', 'test']

		self.config = config
		self.mode = mode.lower()

		self.session = session
		self.embed_dim = config.embed_dim
		self.vocab_dim = config.vocab_dim
		self.fact_dim = config.fact_dim
		self.history_dim = config.history_dim
		self.decod_dim = config.decoder_dim
		self.img_feature_dim = config.img_feature_size
		self.start_token, self.end_token = config.start_token, config.end_token
		self.pad_token = config.pad_token
		self.batch_size = config.batch_size
		self.save_each_epoch = False

		with tf.variable_scope("t_op"):
			self.t_op = tf.Variable(0, trainable=False)
			self.t_add_op = self.t_op.assign_add(1)

		self.use_beamsearch = False
		if self.mode in ['decode', 'rank']:
			self.beam_width = config.beam_width
			self.use_beamsearch = True if self.beam_width > 1 else False
			self.max_decode_step = config.max_decode_step

		self.build_model()

	def build_model(self):
		with tf.variable_scope("abot"):
			self.init_placeholders()
			self.build_encoder()
			self.build_decoder()
			self.build_training()
			self.summary_op = tf.summary.merge_all()

	def init_placeholders(self):
		self.imfeat_ph = tf.placeholder(dtype=tf.float32,
		                                shape=(None, self.img_feature_dim),
		                                name='im_feats')

		self.fact_encoder_inputs = tf.placeholder(dtype=tf.int32,
		                                          shape=(None, None),
		                                          name='fact_encoder_inputs')
		self.fact_encoder_inputs_length = tf.placeholder(dtype=tf.int32,
		                                                 shape=(None,),
		                                                 name='fact_encoder_inputs_length')

		self.ques_encoder_inputs = tf.placeholder(dtype=tf.int32,
		                                          shape=(None, None),
		                                          name='ques_encoder_inputs')

		self.ques_encoder_inputs_length = tf.placeholder(dtype=tf.int32,
		                                                 shape=(None,),
		                                                 name='ques_encoder_inputs_length')

		self.decoder_inputs = tf.placeholder(dtype=tf.int32,
		                                     shape=(None, None),
		                                     name='decoder_inputs')

		self.decoder_inputs_length = tf.placeholder(dtype=tf.int32,
		                                            shape=(None,),
		                                            name='decoder_inputs_length')

		decoder_start_token = tf.ones(shape=(1, self.batch_size),
		                              dtype=tf.int32) * self.start_token
		decoder_pad_token = tf.ones(shape=(1, self.batch_size),
		                            dtype=tf.int32) * self.pad_token

		self.decoder_inputs_train = tf.concat(
			[decoder_start_token, self.decoder_inputs], axis=0
		)
		self.decoder_inputs_length_train = self.decoder_inputs_length + 1

		decoder_train_targets = tf.concat([self.decoder_inputs, decoder_pad_token],
		                                  axis=0)
		decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
		decoder_train_targets_eos_mask = tf.one_hot(self.decoder_inputs_length_train - 1,
		                                            decoder_train_targets_seq_len,
		                                            on_value=self.end_token,
		                                            off_value=self.pad_token,
		                                            dtype=tf.int32)

		decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask,
		                                              [1, 0])
		decoder_train_targets = tf.add(decoder_train_targets,
		                               decoder_train_targets_eos_mask)

		self.decoder_targets_train = decoder_train_targets
		self.c_state_ph = tf.placeholder(dtype=tf.float32,
		                                 shape=(self.batch_size, self.history_dim),
		                                 name='qbot_cell_c1')

		self.h_state_ph = tf.placeholder(dtype=tf.float32,
		                                 shape=(self.batch_size, self.history_dim),
		                                 name='qbot_cell_h1')

		self.c2_state_ph = tf.placeholder(dtype=tf.float32,
		                                  shape=(self.batch_size, self.history_dim),
		                                  name='qbot_cell_c2')

		self.h2_state_ph = tf.placeholder(dtype=tf.float32,
		                                  shape=(self.batch_size, self.history_dim),
		                                  name='qbot_cell_h2')

		self.abot_history_state = tuple([LSTMStateTuple(c=self.c_state_ph, h=self.h_state_ph),
		                                 LSTMStateTuple(c=self.c2_state_ph, h=self.h2_state_ph)])

		sqrt3 = math.sqrt(3)
		initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
		self.embedding_matrix = tf.get_variable(name='embedding_matrix',
		                                        shape=[self.vocab_dim, self.embed_dim],
		                                        initializer=initializer,
		                                        dtype=tf.float32)

	def build_encoder(self):
		print('Building encoder..')
		with tf.variable_scope("encoder"):
			self.fact_encoder_inputs_embedded = tf.nn.embedding_lookup(
				params=self.embedding_matrix, ids=self.fact_encoder_inputs,
				name='fact_embedding_inputs')

			self.ques_encoder_inputs_embedded = tf.nn.embedding_lookup(
				params=self.embedding_matrix, ids=self.ques_encoder_inputs,
				name='ques_embedding_inputs'
			)

			with tf.variable_scope("fact_encoder"):
				self.fact_encoder_cell = MultiRNNCell(
					[LSTMCell(self.fact_dim), LSTMCell(self.fact_dim)])

				self.fact_enc_out, self.fact_enc_state = tf.nn.dynamic_rnn(
					cell=self.fact_encoder_cell, inputs=self.fact_encoder_inputs_embedded,
					sequence_length=self.fact_encoder_inputs_length, dtype=tf.float32,
					time_major=True
				)

			with tf.variable_scope("ques_encoder"):
				self.ques_encoder_cell = MultiRNNCell(
					[LSTMCell(self.fact_dim), LSTMCell(self.fact_dim)])
				self.ques_enc_out, self.ques_enc_state = tf.nn.dynamic_rnn(
					cell=self.ques_encoder_cell, inputs=self.ques_encoder_inputs_embedded,
					sequence_length=self.ques_encoder_inputs_length, dtype=tf.float32,
					time_major=True
				)

			with tf.variable_scope("history_encoder"):
				self.history_encoder_cell = MultiRNNCell(
					[LSTMCell(self.history_dim), LSTMCell(self.history_dim)])

				fact_state = self.fact_enc_state[-1].h
				ques_state = self.ques_enc_state[-1].h

				history_input = tf.concat(values=[fact_state, ques_state, self.imfeat_ph],
				                          axis=1,
				                          name="history_input")
				history_input = tf.expand_dims(history_input, axis=0)
				self.hist_enc_out, self.hist_enc_state = tf.nn.dynamic_rnn(
					cell=self.history_encoder_cell, inputs=history_input,
					initial_state=self.abot_history_state,
					dtype=tf.float32, time_major=True
				)

	def build_decoder(self):
		print('Buidling decoder...')
		with tf.variable_scope("decoder"):
			# Get decoder cell and initial state
			self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()

			# Output projection layer
			output_layer = Dense(self.vocab_dim, name='output_projection')

			if self.mode == 'train':
				# Construct inputs
				self.decoder_inputs_embedded = tf.nn.embedding_lookup(
					self.embedding_matrix,
					self.decoder_inputs_train)

				training_helper = seq2seq.TrainingHelper(
					inputs=self.decoder_inputs_embedded,
					sequence_length=self.decoder_inputs_length_train,
					time_major=True,
					name='training_helper')

				training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
				                                        helper=training_helper,
				                                        initial_state=self.decoder_initial_state,
				                                        output_layer=output_layer)

				# Maximum decoder time_steps in current batch
				max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

				(self.decoder_outputs_train, self.decoder_last_state_train,
				 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
					decoder=training_decoder,
					output_time_major=True,
					impute_finished=True,
					maximum_iterations=max_decoder_length))

				self.decoder_logits_train = tf.identity(
					self.decoder_outputs_train.rnn_output)
				self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
				                                    name='decoder_pred_train')

				self.masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
				                              maxlen=max_decoder_length, dtype=tf.float32,
				                              name='masks')
			elif self.mode in ['decode', 'rank']:
				start_tokens = tf.ones([self.batch_size, ],
				                       tf.int32) * self.start_token
				end_token = self.end_token
				if not self.use_beamsearch:
					# Greedy decoder
					decoder_helper = seq2seq.GreedyEmbeddingHelper(
						start_tokens=start_tokens,
						end_token=end_token,
						embedding=self.embedding_matrix)

					print('building greedy decoder...')
					inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
					                                         helper=decoder_helper,
					                                         initial_state=self.decoder_initial_state,
					                                         output_layer=output_layer)
				else:
					print('building beam search decoder...')
					inference_decoder = beam_search_decoder.BeamSearchDecoder(
						cell=self.decoder_cell,
						embedding=self.embedding_matrix,
						start_tokens=start_tokens,
						end_token=end_token,
						initial_state=self.decoder_initial_state,
						beam_width=self.beam_width,
						output_layer=output_layer)

				(self.decoder_outputs_decode, self.decoder_last_state_decode,
				 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
					decoder=inference_decoder,
					output_time_major=True,
					maximum_iterations=self.max_decode_step))

				if not self.use_beamsearch:
					# shape is [max_steps, batch_size]
					self.decoder_pred_decode = tf.expand_dims(
						self.decoder_outputs_decode.sample_id, axis=-1)
					self.decoder_outputs_length_decode = tf.expand_dims(
						self.decoder_outputs_length_decode, axis=-1
					)
				else:
					# shape is [max_steps, batch_size, beam_width]
					self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids

	def build_training(self):
		if self.mode == 'train':
			print('Building training ops...')

			# Seq2Seq training
			self.loss = seq2seq.sequence_loss(
				logits=tf.transpose(self.decoder_logits_train, [1, 0, 2]),
				targets=tf.transpose(self.decoder_targets_train, [1, 0]),
				weights=self.masks,
				average_across_batch=True,
				average_across_timesteps=True)

			tf.summary.scalar('loss', self.loss)
			self.optimizer = tf.train.AdamOptimizer()
			grads_vars = self.optimizer.compute_gradients(self.loss)
			cliped_gradients = [(tf.clip_by_value(grad, -5., 5.), tvar) for grad, tvar in
			                    grads_vars if grad is not None]
			self.update_op = self.optimizer.apply_gradients(cliped_gradients, self.t_op)

	def build_decoder_cell(self):
		encoder_last_state = self.hist_enc_state
		if self.use_beamsearch:
			print("use beam search decoding..")
			encoder_last_state = nest.map_structure(
				lambda s: seq2seq.tile_batch(s, self.beam_width), encoder_last_state
			)

		decoder_initial_state = encoder_last_state
		decoder_cell = MultiRNNCell([LSTMCell(self.decod_dim), LSTMCell(self.decod_dim)])

		return decoder_cell, decoder_initial_state

	def save(self, path, var_list=None, global_step=None):
		# var_list = None returns the list of all saveable variables
		sess = self.session
		saver = tf.train.Saver(var_list)

		# temporary code
		save_path = saver.save(sess, save_path=path, global_step=global_step)
		print('model saved at %s' % save_path)

	def restore(self, sess, path, var_list=None):
		# var_list = None returns the list of all saveable variables
		self.session.run(tf.global_variables_initializer())
		saver = tf.train.Saver(var_list)
		saver.restore(sess, save_path=path)
		print('model restored from %s' % path)

	def get_batch_inputs(self, batch, round):
		q_len = batch['question_lengths'][:, round]
		h_len = batch['history_lengths'][:, round]
		a_len = batch['answer_lengths'][:, round]
		q = batch['question'][0:int(np.max(q_len)), round, :]
		h = batch['history'][0:int(np.max(h_len)), round, :]
		a = batch['answer'][0:int(np.max(a_len)), round, :]

		return q, h, a, q_len, h_len, a_len, batch['img_feats'], batch['img_inds']

	def make_train_feed(self, data, c1, h1, c2, h2):
		question, history, answer, q_len, h_len, a_len, img_feats, img_inds = data
		return {
			self.fact_encoder_inputs: history,
			self.fact_encoder_inputs_length: h_len,
			self.ques_encoder_inputs: question,
			self.ques_encoder_inputs_length: q_len,
			self.decoder_inputs: answer,
			self.decoder_inputs_length: a_len,
			self.c_state_ph: c1,
			self.h_state_ph: h1,
			self.c2_state_ph: c2,
			self.h2_state_ph: h2,
			self.imfeat_ph: img_feats  # TODO(Mircea): Check correctness
		}

	def make_decode_feed(self, data, c1, h1, c2, h2):
		question, history, aanswer, q_len, h_len, a_len, img_feats, img_inds = data
		print('Question received by abot is {} with shape {}'.format(question, question.shape))
		return {
			self.fact_encoder_inputs: history,
			self.fact_encoder_inputs_length: h_len,
			self.ques_encoder_inputs: question,
			self.ques_encoder_inputs_length: q_len,
			self.c_state_ph: c1,
			self.h_state_ph: h1,
			self.c2_state_ph: c2,
			self.h2_state_ph: h2,
			self.imfeat_ph: img_feats
		}

	def make_true_decode_feed(self, history, history_length, question, question_length, img_feats,
	                          c1, h1, c2, h2):
		return {
			self.fact_encoder_inputs: history,
			self.fact_encoder_inputs_length: history_length,
			self.ques_encoder_inputs: question,
			self.ques_encoder_inputs_length: question_length,
			self.c_state_ph: c1,
			self.h_state_ph: h1,
			self.c2_state_ph: c2,
			self.h2_state_ph: h2,
			self.imfeat_ph: img_feats
		}

	def train(self, data, epochs):
		start_time = time.time()
		print('Started training ABOT model for {} epochs.'.format(epochs))
		num_batches = int(np.ceil(data.num_train_threads / self.batch_size))

		self.log_writer = tf.summary.FileWriter(self.config.logs_path,
		                                        graph=self.session.graph)
		self.session.run(tf.global_variables_initializer())
		for cur_epoch in range(epochs):
			for cur_batch in range(num_batches):
				batch, _ = data.get_train_batch(self.batch_size, time_major=True)
				c1 = np.zeros((self.batch_size, self.history_dim))
				h1 = np.zeros((self.batch_size, self.history_dim))
				c2 = np.zeros((self.batch_size, self.history_dim))
				h2 = np.zeros((self.batch_size, self.history_dim))
				batch_loss = 0.
				# batch_regression_loss = 0.
				batch_start_time = time.time()
				for cur_round in range(10):
					feed_dict = self.make_train_feed(
						data=self.get_batch_inputs(batch, cur_round),
						c1=c1,
						h1=h1,
						c2=c2,
						h2=h2)

					fetches = [self.hist_enc_state, self.loss, self.update_op]
					if cur_round % 5 == 0 and cur_batch % 50 == 0:
						fetches += [self.summary_op]
						states, round_loss, _, summ = self.session.run(fetches, feed_dict)
						self.log_writer.add_summary(summ, self.t_op.eval())
					else:
						states, round_loss, _ = self.session.run(fetches, feed_dict)
					c1, h1 = states[0].c, states[0].h
					c2, h2 = states[1].c, states[1].h
					batch_loss += round_loss

				batch_duration = time.time() - batch_start_time
				logger.record_tabular('Time elapsed', time.time() - start_time)
				logger.record_tabular('Batch duration', batch_duration)
				logger.record_tabular('(Batch, Total)', (cur_batch, num_batches))
				logger.record_tabular('Epoch ', cur_epoch)
				logger.record_tabular('Batch loss ', batch_loss / 10.)
				logger.dump_tabular()

			if self.save_each_epoch:
				save_path = os.path.join(self.config.save_path,
				                         'epoch_{}'.format(cur_epoch), 'model.ckpt')
				self.save(save_path)
			logger.log('Finished epoch {}/{}'.format(cur_epoch, epochs))

		self.log_writer.close()
		save_path = os.path.join(self.config.save_path, self.config.model_name,
		                         'model.ckpt')
		self.save(save_path)

	def decode(self, data):
		vocabulary = data.data['ind2word']
		batch, _, _ = data.get_test_batch(np.random.randint(0, 40000), self.batch_size,
		                                  time_major=True)
		c1 = np.zeros((self.batch_size, self.history_dim))
		h1 = np.zeros((self.batch_size, self.history_dim))
		c2 = np.zeros((self.batch_size, self.history_dim))
		h2 = np.zeros((self.batch_size, self.history_dim))
		print("caption: {}".format(" ".join(list(
			vocabulary[token] for token in batch['history'][:, 0, 0] if
			token in vocabulary))))
		for cur_round in range(10):
			feed_dict = self.make_decode_feed(
				data=self.get_batch_inputs(batch, cur_round),
				c1=c1,
				h1=h1,
				c2=c2,
				h2=h2
			)
			fetches = [self.hist_enc_state, self.decoder_pred_decode,
			           self.decoder_outputs_length_decode]
			states, decoding, decodings_length = self.session.run(fetches, feed_dict=feed_dict)
			c1, h1 = states[0].c, states[0].h
			c2, h2 = states[1].c, states[1].h
			self.print_greedy_dround(decoding[:, :, 0], decodings_length, vocabulary,
			                         batch['question'][:, cur_round, 0])

			if cur_round == 5:
				np.save('abot_decoding.npy', decoding)
				np.save('abot_decoding_length.npy', decodings_length)

	def print_greedy_dround(self, decoding, decoding_length, vocabulary, question):
		print('Decoding for all batches is {}'.format(decoding))
		# decoding to [batch_size, time_steps]
		decoding = np.transpose(decoding)[0]
		print('Decoding shape is {}, question shape is {}'.format(decoding.shape, question.shape))
		print('Decoding raw is {}'.format(decoding))
		print('Question raw is {}'.format(question))

		print('Decoding length is {}'.format(decoding_length))
		print('Decoding length shape is {}'.format(decoding_length.shape))

		print('Question is')
		print(' '.join(
			list(vocabulary[token] for token in question if token in vocabulary)))
		to_print = list(vocabulary[token] for token in decoding if token in vocabulary)
		print('List to be printed is length {}'.format(len(to_print)))
		print(" ".join(to_print))
		print("----------")


# def preprocess_qa(questions, answers, stop_token=0):
# 	# Shapes
# 	sequence_length = questions.shape[0] + answers.shape[0]
# 	batch_size = questions.shape[1]
#
# 	facts = np.zeros(shape=(sequence_length, batch_size))
# 	facts[:questions.shape[0], :questions.shape[1]] = questions
# 	fact_len = np.zeros(batch_size)
#
# 	# transpose facts so that each datapoint is a row
# 	for i, row in enumerate(facts.T):
#
# 		# will always exist as matrix is initialized to zeros
# 		first_zero_idx = min(np.argmax(row == stop_token), np.argmax(row == 0))
#
# 		# indices where answers are zeros
# 		answer_zeros = np.where(answers[:, i] == 0)[0]
#
# 		if len(answer_zeros) > 0:
# 			answer_len = np.min(answer_zeros) + 1  # correction for 0-based indexing
# 		else:  # maximal size
# 			answer_len = answers.shape[0]
#
# 		# Remove stop token
# 		if answers[answer_len - 1, i] == stop_token:
# 			answer_len -= 1
#
# 		# Assign
# 		facts[first_zero_idx:first_zero_idx + answer_len, i] = answers[:answer_len, i]
#
# 		# Lengths
# 		question_len = first_zero_idx + 1
# 		fact_len[i] = question_len + answer_len
#
# 	return facts, fact_len

class QBOT(object):
	def __init__(self,
	             session,
	             config,
	             mode):

		assert mode.lower() in ['train', 'decode', 'rank', 'test']

		self.config = config
		self.mode = mode.lower()

		self.session = session
		self.embed_dim = config.embed_dim
		self.vocab_dim = config.vocab_dim
		self.fact_dim = config.fact_dim
		self.history_dim = config.history_dim
		self.decod_dim = config.decoder_dim
		self.img_feature_dim = config.img_feature_size
		self.start_token, self.end_token = config.start_token, config.end_token
		self.pad_token = config.pad_token
		self.batch_size = config.batch_size
		self.save_each_epoch = False

		with tf.variable_scope("t_op"):
			self.t_op = tf.Variable(0, trainable=False)
			self.t_add_op = self.t_op.assign_add(1)

		self.use_beamsearch = False
		if self.mode in ['decode', 'rank']:
			self.beam_width = config.beam_width
			self.use_beamsearch = True if self.beam_width > 1 else False
			self.max_decode_step = config.max_decode_step

		self.build_model()

	def build_model(self):
		self.init_placeholders()
		self.build_encoder()
		self.build_regression()
		self.build_decoder()
		self.build_training()
		self.summary_op = tf.summary.merge_all()

	def init_placeholders(self):
		print('Building placeholders...')

		# Regression placeholders
		self.imfeat_ph = tf.placeholder(dtype=tf.float32,
		                                shape=(None, self.img_feature_dim),
		                                name='im_feats')

		# Seq2Seq placeholders
		self.encoder_inputs = tf.placeholder(dtype=tf.int32,
		                                     shape=(None, None),
		                                     name='encoder_inputs')
		self.encoder_inputs_length = tf.placeholder(dtype=tf.int32,
		                                            shape=(None,),
		                                            name='encoder_inputs_length')

		self.decoder_inputs = tf.placeholder(dtype=tf.int32,
		                                     shape=(None, None),
		                                     name='decoder_inputs')

		self.decoder_inputs_length = tf.placeholder(dtype=tf.int32,
		                                            shape=(None,),
		                                            name='decoder_inputs_length')

		decoder_start_token = tf.ones(shape=(1, self.batch_size),
		                              dtype=tf.int32) * self.start_token
		decoder_pad_token = tf.ones(shape=(1, self.batch_size),
		                            dtype=tf.int32) * self.pad_token

		self.decoder_inputs_train = tf.concat(
			[decoder_start_token, self.decoder_inputs], axis=0
		)
		self.decoder_inputs_length_train = self.decoder_inputs_length + 1

		decoder_train_targets = tf.concat([self.decoder_inputs, decoder_pad_token],
		                                  axis=0)
		decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
		decoder_train_targets_eos_mask = tf.one_hot(self.decoder_inputs_length_train - 1,
		                                            decoder_train_targets_seq_len,
		                                            on_value=self.end_token,
		                                            off_value=self.pad_token,
		                                            dtype=tf.int32)

		decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask,
		                                              [1, 0])
		decoder_train_targets = tf.add(decoder_train_targets,
		                               decoder_train_targets_eos_mask)

		self.decoder_targets_train = decoder_train_targets
		self.c_state_ph = tf.placeholder(dtype=tf.float32,
		                                 shape=(self.batch_size, self.history_dim),
		                                 name='qbot_cell_c1')

		self.h_state_ph = tf.placeholder(dtype=tf.float32,
		                                 shape=(self.batch_size, self.history_dim),
		                                 name='qbot_cell_h1')

		self.c2_state_ph = tf.placeholder(dtype=tf.float32,
		                                  shape=(self.batch_size, self.history_dim),
		                                  name='qbot_cell_c2')

		self.h2_state_ph = tf.placeholder(dtype=tf.float32,
		                                  shape=(self.batch_size, self.history_dim),
		                                  name='qbot_cell_h2')

		self.qbot_history_state = tuple([LSTMStateTuple(c=self.c_state_ph, h=self.h_state_ph),
		                                 LSTMStateTuple(c=self.c2_state_ph, h=self.h2_state_ph)])

		sqrt3 = math.sqrt(3)
		initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
		self.embedding_matrix = tf.get_variable(name='embedding_matrix',
		                                        shape=[self.vocab_dim, self.embed_dim],
		                                        initializer=initializer,
		                                        dtype=tf.float32)

	def build_encoder(self):
		print('Building encoder..')
		with tf.variable_scope("encoder"):
			self.encoder_inputs_embedded = tf.nn.embedding_lookup(
				params=self.embedding_matrix, ids=self.encoder_inputs,
				name='encoder_embedding_inputs')

			with tf.variable_scope("fact_encoder"):
				self.fact_encoder_cell = MultiRNNCell(
					[LSTMCell(self.fact_dim), LSTMCell(self.fact_dim)])

				self.fact_enc_out, self.fact_enc_state = tf.nn.dynamic_rnn(
					cell=self.fact_encoder_cell, inputs=self.encoder_inputs_embedded,
					sequence_length=self.encoder_inputs_length, dtype=tf.float32,
					time_major=True
				)

			with tf.variable_scope("history_encoder"):
				self.history_encoder_cell = MultiRNNCell(
					[LSTMCell(self.history_dim), LSTMCell(self.history_dim)])
				history_input = tf.expand_dims(self.fact_enc_state[-1].h, axis=0)
				self.hist_enc_out, self.hist_enc_state = tf.nn.dynamic_rnn(
					cell=self.history_encoder_cell, inputs=history_input,
					initial_state=self.qbot_history_state,
					dtype=tf.float32, time_major=True
				)

	def build_regression(self):
		print('Building regression...')
		encoder_state = self.hist_enc_state[-1].h
		encoder_state_shape = encoder_state.get_shape()[-1].value
		self.rw = tf.get_variable("prediction_w",
		                          shape=(encoder_state_shape, self.img_feature_dim))
		self.rb = tf.get_variable("prediction_b",
		                          shape=(self.img_feature_dim,))
		self.y_t = tf.matmul(encoder_state, self.rw) + self.rb

	def build_decoder(self):
		print('Buidling decoder...')
		with tf.variable_scope("decoder"):
			# Get decoder cell and initial state
			self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()

			# Output projection layer
			output_layer = Dense(self.vocab_dim, name='output_projection')

			if self.mode == 'train':
				# Construct inputs
				self.decoder_inputs_embedded = tf.nn.embedding_lookup(
					self.embedding_matrix,
					self.decoder_inputs_train)

				training_helper = seq2seq.TrainingHelper(
					inputs=self.decoder_inputs_embedded,
					sequence_length=self.decoder_inputs_length_train,
					time_major=True,
					name='training_helper')

				training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
				                                        helper=training_helper,
				                                        initial_state=self.decoder_initial_state,
				                                        output_layer=output_layer)

				# Maximum decoder time_steps in current batch
				max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

				(self.decoder_outputs_train, self.decoder_last_state_train,
				 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
					decoder=training_decoder,
					output_time_major=True,
					impute_finished=True,
					maximum_iterations=max_decoder_length))

				self.decoder_logits_train = tf.identity(
					self.decoder_outputs_train.rnn_output)
				self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
				                                    name='decoder_pred_train')

				self.masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
				                              maxlen=max_decoder_length, dtype=tf.float32,
				                              name='masks')
			elif self.mode in ['decode', 'rank']:
				start_tokens = tf.ones([self.batch_size, ],
				                       tf.int32) * self.start_token
				end_token = self.end_token
				if not self.use_beamsearch:
					# Greedy decoder
					decoder_helper = seq2seq.GreedyEmbeddingHelper(
						start_tokens=start_tokens,
						end_token=end_token,
						embedding=self.embedding_matrix)

					print('building greedy decoder...')
					inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
					                                         helper=decoder_helper,
					                                         initial_state=self.decoder_initial_state,
					                                         output_layer=output_layer)

				else:
					print('building beam search decoder...')
					inference_decoder = beam_search_decoder.BeamSearchDecoder(
						cell=self.decoder_cell,
						embedding=self.embedding_matrix,
						start_tokens=start_tokens,
						end_token=end_token,
						initial_state=self.decoder_initial_state,
						beam_width=self.beam_width,
						output_layer=output_layer)

				(self.decoder_outputs_decode, self.decoder_last_state_decode,
				 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
					decoder=inference_decoder,
					output_time_major=True,
					maximum_iterations=self.max_decode_step))

				if not self.use_beamsearch:
					# shape is [max_steps, batch_size]
					self.decoder_pred_decode = tf.expand_dims(
						self.decoder_outputs_decode.sample_id, axis=-1)
					self.decoder_outputs_length_decode = tf.expand_dims(
						self.decoder_outputs_length_decode, axis=-1
					)
				else:
					# shape is [max_steps, batch_size, beam_width]
					# note: The final computation uses GatherTree to identify the true indices at each time
					# step. This means that the 0th beam is the one with the highest score; and
					# you should be able to use predicted_ids[:, :, 0] to access it.
					self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids

	def build_training(self):
		self.optimizer = tf.train.AdamOptimizer()
		if self.mode == 'train':
			print('Building training ops...')
			# Seq2Seq training
			self.loss = seq2seq.sequence_loss(
				logits=tf.transpose(self.decoder_logits_train, [1, 0, 2]),
				targets=tf.transpose(self.decoder_targets_train, [1, 0]),
				weights=self.masks,
				average_across_batch=True,
				average_across_timesteps=True)

			tf.summary.scalar('loss', self.loss)
			grads_vars = self.optimizer.compute_gradients(self.loss)
			cliped_gradients = [(tf.clip_by_value(grad, -5., 5.), tvar) for grad, tvar in
			                    grads_vars if grad is not None]
			self.update_op = self.optimizer.apply_gradients(cliped_gradients, self.t_op)

		# Regression training
		# self.l2_dist_sq = tf.sqrt(tf.reduce_sum(tf.square(self.y_t - self.imfeat_ph),
		#                                         name='prediction_l2'))
		# self.l2_dist_sq = tf.reduce_sum(tf.square(self.y_t - self.imfeat_ph),
		#                                         name='prediction_l2')
		self.l2_dist_sq = tf.sqrt(
			tf.reduce_sum(tf.square(self.y_t - self.imfeat_ph), axis=1))
		self.batch_l2_loss = tf.reduce_mean(self.l2_dist_sq)

		mse_grads_vars = self.optimizer.compute_gradients(self.batch_l2_loss)
		clipped_gradients_regression = [(tf.clip_by_value(grad, -5., 5.), tvar) for
		                                grad, tvar in
		                                mse_grads_vars if grad is not None]
		tf.summary.scalar('l2_dist_batch', self.batch_l2_loss)
		self.update_pred_op = self.optimizer.apply_gradients(clipped_gradients_regression,
		                                                     self.t_op)

	def build_decoder_cell(self):
		encoder_last_state = self.hist_enc_state
		if self.use_beamsearch:
			print("use beam search decoding..")
			encoder_last_state = nest.map_structure(
				lambda s: seq2seq.tile_batch(s, self.beam_width), encoder_last_state
			)

		decoder_initial_state = encoder_last_state
		decoder_cell = MultiRNNCell([LSTMCell(self.decod_dim), LSTMCell(self.decod_dim)])

		return decoder_cell, decoder_initial_state

	def save(self, path, var_list=None, global_step=None):
		# var_list = None returns the list of all saveable variables
		sess = self.session
		saver = tf.train.Saver(var_list)

		# temporary code
		save_path = saver.save(sess, save_path=path, global_step=global_step)
		print('model saved at %s' % save_path)

	def restore(self, sess, path, var_list=None):
		# var_list = None returns the list of all saveable variables
		self.session.run(tf.global_variables_initializer())
		saver = tf.train.Saver(var_list)
		saver.restore(sess, save_path=path)
		print('model restored from %s' % path)

	def get_batch_inputs(self, batch, round):
		q_len = batch['question_lengths'][:, round]
		h_len = batch['history_lengths'][:, round]
		a_len = batch['answer_lengths'][:, round]
		q = batch['question'][0:int(np.max(q_len)), round, :]
		h = batch['history'][0:int(np.max(h_len)), round, :]
		a = batch['answer'][0:int(np.max(a_len)), round, :]

		return q, h, a, q_len, h_len, a_len, batch['img_feats'], batch['img_inds']

	def make_train_feed(self, data, c1, h1, c2, h2):
		question, history, aanswer, q_len, h_len, a_len, img_feats, img_inds = data
		return {
			self.encoder_inputs: history,
			self.encoder_inputs_length: h_len,
			self.decoder_inputs: question,
			self.decoder_inputs_length: q_len,
			self.c_state_ph: c1,
			self.h_state_ph: h1,
			self.c2_state_ph: c2,
			self.h2_state_ph: h2,
			self.imfeat_ph: img_feats  # TODO(Mircea): Check correctness
		}

	def make_decode_feed(self, data, c1, h1, c2, h2):
		question, history, aanswer, q_len, h_len, a_len, img_feats, img_inds = data
		return {
			self.encoder_inputs: history,
			self.encoder_inputs_length: h_len,
			self.c_state_ph: c1,
			self.h_state_ph: h1,
			self.c2_state_ph: c2,
			self.h2_state_ph: h2,
		}

	def make_true_decode_feed(self, history, history_length, c1, h1, c2, h2):
		return {
			self.encoder_inputs: history,
			self.encoder_inputs_length: history_length,
			self.c_state_ph: c1,
			self.h_state_ph: h1,
			self.c2_state_ph: c2,
			self.h2_state_ph: h2,
		}

	def make_rank_feed(self, data, c1, h1, c2, h2):
		question, history, aanswer, q_len, h_len, a_len, img_feats, img_inds = data
		return {
			self.encoder_inputs: history,
			self.encoder_inputs_length: h_len,
			self.c_state_ph: c1,
			self.h_state_ph: h1,
			self.c2_state_ph: c2,
			self.h2_state_ph: h2,
			self.imfeat_ph: img_feats
		}

	def train(self, data, epochs):
		start_time = time.time()
		print('Started training model for {} epochs.'.format(epochs))
		num_batches = int(np.ceil(data.num_train_threads / self.batch_size))

		self.log_writer = tf.summary.FileWriter(self.config.logs_path,
		                                        graph=self.session.graph)
		self.session.run(tf.global_variables_initializer())
		for cur_epoch in range(epochs):
			for cur_batch in range(num_batches):
				batch, _ = data.get_train_batch(self.batch_size, time_major=True)
				c1 = np.zeros((self.batch_size, self.history_dim))
				h1 = np.zeros((self.batch_size, self.history_dim))
				c2 = np.zeros((self.batch_size, self.history_dim))
				h2 = np.zeros((self.batch_size, self.history_dim))
				batch_loss = 0.
				batch_regression_loss = 0.
				batch_start_time = time.time()
				for cur_round in range(10):
					feed_dict = self.make_train_feed(
						data=self.get_batch_inputs(batch, cur_round),
						c1=c1,
						h1=h1,
						c2=c2,
						h2=h2
					)

					fetches = [self.hist_enc_state, self.loss, self.batch_l2_loss,
					           self.update_op, self.update_pred_op]
					# fetches = [self.hist_enc_state, self.loss, self.update_op]
					if cur_round % 5 == 0 and cur_batch % 50 == 0:
						fetches += [self.summary_op]
						states, round_loss, mse, _, _, summ = self.session.run(fetches,
						                                                       feed_dict)
						# states, round_loss, _, summ = self.session.run(fetches, feed_dict)
						self.log_writer.add_summary(summ, self.t_op.eval())
					else:
						states, round_loss, mse, _, _ = self.session.run(fetches,
						                                                 feed_dict)
					c1, h1 = states[0].c, states[0].h
					c2, h2 = states[1].c, states[1].h
					batch_loss += round_loss
					batch_regression_loss += mse

				batch_duration = time.time() - batch_start_time
				logger.record_tabular('Time elapsed', time.time() - start_time)
				logger.record_tabular('Batch duration', batch_duration)
				logger.record_tabular('(Batch, Total)', (cur_batch, num_batches))
				logger.record_tabular('Epoch ', cur_epoch)
				logger.record_tabular('Batch loss ', batch_loss / 10.)
				logger.record_tabular('Batch l2_dist_sq loss ',
				                      batch_regression_loss / 10.)
				logger.dump_tabular()

			if self.save_each_epoch:
				save_path = os.path.join(self.config.save_path,
				                         'epoch_{}'.format(cur_epoch), 'model.ckpt')
				self.save(save_path)
			logger.log('Finished epoch {}/{}'.format(cur_epoch, epochs))

		self.log_writer.close()
		save_path = os.path.join(self.config.save_path, self.config.model_name,
		                         'model.ckpt')
		self.save(save_path)

	def decode(self, data):
		vocabulary = data.data['ind2word']
		batch, _, _ = data.get_test_batch(np.random.randint(0, 39999), self.batch_size,
		                                  time_major=True)
		c1 = np.zeros((self.batch_size, self.history_dim))
		h1 = np.zeros((self.batch_size, self.history_dim))
		c2 = np.zeros((self.batch_size, self.history_dim))
		h2 = np.zeros((self.batch_size, self.history_dim))
		print("caption: {}".format(" ".join(list(
			vocabulary[token] for token in batch['history'][:, 0, 0] if
			token in vocabulary))))
		print('Example first history fact: {}'.format(batch['history'][:, 1, 0]))
		print('Text: {}'.format(" ".join(
			list(vocabulary[token] for token in batch['history'][:, 1, 0] if token in vocabulary))))
		for cur_round in range(10):
			feed_dict = self.make_decode_feed(
				data=self.get_batch_inputs(batch, cur_round),
				c1=c1,
				h1=h1,
				c2=c2,
				h2=h2
			)
			fetches = [self.hist_enc_state, self.decoder_pred_decode,
			           self.decoder_outputs_length_decode]
			states, decoding, decoding_length = self.session.run(fetches, feed_dict=feed_dict)
			c1, h1 = states[0].c, states[0].h
			c2, h2 = states[1].c, states[1].h
			self.print_greedy_dround(decoding[:, :, 0], decoding_length[:, 0], vocabulary)

	def print_greedy_dround(self, decoding, decoding_length, vocabulary):
		# decoding to [batch_size, time_steps]
		print('Decoding shape is {}'.format(decoding.shape))
		print('Decoding raw is {}'.format(decoding))
		decoding = np.transpose(decoding)[0]

		print('Decoding length is {}'.format(decoding_length))
		print('Decoding length shape is {}'.format(decoding_length.shape))

		# print("Raw decoding is {}".format(decoding))
		# print("It is a vector length {}".format(decoding.shape))
		to_print = list(vocabulary[token] for token in decoding if token in vocabulary)
		print('List to be printed is length {}'.format(len(to_print)))
		print(" ".join(to_print))
		print("----------")

	def rank(self, data, eval_size=10000):
		print('Started ranking...')
		assert eval_size < 40000
		# Get test set to evaluate on
		val_images, val_indices = data.data['val_img_fv'][:eval_size], data.data['val_img_pos'][
		                                                               :eval_size]
		batch_size = self.batch_size
		all_ranks = []
		for cur_batch in range(0, eval_size, batch_size):
			if cur_batch % 1000 == 0:
				print('Ranking at batch, ', cur_batch)
			batch, *_ = data.get_test_batch(start_id=cur_batch, batch_size=batch_size)
			c1 = np.zeros((self.batch_size, self.history_dim))
			h1 = np.zeros((self.batch_size, self.history_dim))
			c2 = np.zeros((self.batch_size, self.history_dim))
			h2 = np.zeros((self.batch_size, self.history_dim))
			for cur_round in range(10):
				feed_dict = self.make_rank_feed(
					data=self.get_batch_inputs(batch, cur_round),
					c1=c1,
					h1=h1,
					c2=c2,
					h2=h2
				)
				fetches = [self.hist_enc_state]
				if cur_round == 9:
					fetches += [self.y_t]
					states, prediction = self.session.run(fetches, feed_dict=feed_dict)
				else:
					states = self.session.run(fetches, feed_dict=feed_dict)[0]
				c1, h1 = states[0].c, states[0].h
				c2, h2 = states[1].c, states[1].h

			# Get ranking for this batch
			batch_ranks = ranks(prediction, val_images, batch['img_inds'])
			all_ranks.extend(batch_ranks)

		scores = list(range(eval_size))
		percentiles = list(
			map(lambda rank: stats.percentileofscore(scores, eval_size - rank), all_ranks))

		print('Mean percentile ranks is {}'.format(np.mean(percentiles)))
		print('Mean rank is {}'.format(np.mean(all_ranks)))

	def test(self, data):
		vocabulary = data.data['ind2word']
		batch, _ = data.get_train_batch(self.batch_size, time_major=True)
		c1 = np.zeros((self.batch_size, self.history_dim))
		h1 = np.zeros((self.batch_size, self.history_dim))
		c2 = np.zeros((self.batch_size, self.history_dim))
		h2 = np.zeros((self.batch_size, self.history_dim))
		for cur_round in range(10):
			feed_dict = self.make_train_feed(
				data=self.get_batch_inputs(batch, cur_round),
				c1=c1,
				h2=h2,
				c2=c2,
				h1=h1
			)
			fetches = [self.decoder_inputs, self.decoder_inputs_train,
			           self.decoder_targets_train]
			di, dit, dtt = self.session.run(fetches, feed_dict=feed_dict)
			print('Decoder inputs for roudn {} are {}'.format(cur_round, di))
			print('Decoder inputs train for round {} are {}'.format(cur_round, dit))
			print('Decoder targets train for round {} are {}'.format(cur_round, dtt))


def decode_both(qbot, abot, data, qbot_session, abot_session, config):
	vocabulary = data.data['ind2word']
	dataset = 'val' if config.dataset == '09' else 'test'
	ub = 39999 if config.dataset == '09' else 9500
	batch, _, _ = data.get_test_batch(np.random.randint(0, ub), config.batch_size,
	                                  time_major=True, subset=dataset)
	c1_q = np.zeros((config.batch_size, config.history_dim))
	h1_q = np.zeros((config.batch_size, config.history_dim))
	c2_q = np.zeros((config.batch_size, config.history_dim))
	h2_q = np.zeros((config.batch_size, config.history_dim))

	c1_a = np.zeros((config.batch_size, config.history_dim))
	h1_a = np.zeros((config.batch_size, config.history_dim))
	c2_a = np.zeros((config.batch_size, config.history_dim))
	h2_a = np.zeros((config.batch_size, config.history_dim))

	# Caption
	qa = batch['history'][:, 0, :]
	qa_len = batch['history_lengths'][:, 0]
	image = batch['img_feats']
	num_rounds = 10

	print('Caption is ')
	print(' '.join(list(vocabulary[token] for token in qa[:, 0] if token in vocabulary)))

	q_concat, q_concat_len = None, None
	for cur_round in range(num_rounds):
		# Obtain question and process it
		qbot_feed = qbot.make_true_decode_feed(qa, qa_len, c1_q, h1_q, c2_q, h2_q)
		qbot_fetches = [qbot.hist_enc_state, qbot.decoder_pred_decode,
		                qbot.decoder_outputs_length_decode]
		qbot_states, qbot_decoding, qbot_decoding_length = qbot_session.run(qbot_fetches,
		                                                                    feed_dict=qbot_feed)
		q, q_len = preprocess_q(qbot_decoding, qbot_decoding_length, config.end_token)

		if cur_round == 0:
			q_concat = q
			q_concat_len = q_len + 1
		else:
			q_concat, q_concat_len = concat_q(q_concat, q_concat_len, qbot_decoding,
			                                  qbot_decoding_length, config.end_token)

		# Store qbot states
		c1_q, h1_q = qbot_states[0].c, qbot_states[0].h
		c2_q, h2_q = qbot_states[1].c, qbot_states[1].h

		# print('Decoded question for round {} batch 0 is '.format(cur_round))
		# print(' '.join(list(vocabulary[token] for token in q[:, 0] if token in vocabulary)))

		# Obtain answer
		abot_feed = abot.make_true_decode_feed(qa, qa_len, q, q_len, image, c1_a, h1_a, c2_a, h2_a)
		abot_fetches = [abot.hist_enc_state, abot.decoder_pred_decode,
		                abot.decoder_outputs_length_decode]
		abot_states, abot_decoding, abot_decoding_length = abot_session.run(abot_fetches,
		                                                                    feed_dict=abot_feed)

		# Store abot states
		c1_a, h1_a = abot_states[0].c, abot_states[0].h
		c2_a, h2_a = abot_states[1].c, abot_states[1].h

		# Concatenate qa
		qa, qa_len = concatenate_qa(qbot_decoding, abot_decoding, qbot_decoding_length,
		                            abot_decoding_length, config.end_token)

		print('Decoded dialong for round {}, batch 0:'.format(cur_round))
		print(' '.join(list(vocabulary[token] for token in qa[:, 0] if token in vocabulary)))

	# print('concatenated q is {}'.format(q_concat))
	# print('concatenated q halfed is {}'.format(q_concat[:np.max(q_concat_len), :]))
	# print('lengths are {}'.format(q_concat_len))


def rank_both(qbot, abot, data, qbot_session, abot_session, config, eval_size=10000):
	print('Started ranking...')
	assert eval_size <= 40000

	if config.dataset == '09':
		val_images = data.data['val_img_fv'][:eval_size]
		val_indices = data.data['val_img_pos'][:eval_size]
	else:
		val_images = data.data['test_img_fv'][:eval_size]
		val_indices = data.data['test_img_pos'][:eval_size]

	dataset = 'val' if config.dataset == '09' else 'test'
	num_rounds = 10
	batch_size = config.batch_size
	all_ranks = []
	for cur_batch in range(0, eval_size, batch_size):
		if cur_batch % 1000 == 0:
			print('Ranking at batch, ', cur_batch)
		batch, *_ = data.get_test_batch(start_id=cur_batch, batch_size=batch_size, subset=dataset)

		c1_q = np.zeros((config.batch_size, config.history_dim))
		h1_q = np.zeros((config.batch_size, config.history_dim))
		c2_q = np.zeros((config.batch_size, config.history_dim))
		h2_q = np.zeros((config.batch_size, config.history_dim))

		c1_a = np.zeros((config.batch_size, config.history_dim))
		h1_a = np.zeros((config.batch_size, config.history_dim))
		c2_a = np.zeros((config.batch_size, config.history_dim))
		h2_a = np.zeros((config.batch_size, config.history_dim))

		qa = batch['history'][:, 0, :]
		qa_len = batch['history_lengths'][:, 0]
		image = batch['img_feats']

		q_concat, q_concat_len = None, None
		for cur_round in range(num_rounds):
			# Obtain question and process it
			qbot_feed = qbot.make_true_decode_feed(qa, qa_len, c1_q, h1_q, c2_q, h2_q)
			qbot_fetches = [qbot.hist_enc_state, qbot.decoder_pred_decode,
			                qbot.decoder_outputs_length_decode]

			qbot_states, qbot_decoding, qbot_decoding_length = qbot_session.run(qbot_fetches,
			                                                                    feed_dict=qbot_feed)
			q, q_len = preprocess_q(qbot_decoding, qbot_decoding_length, config.end_token)

			if cur_round == 0:
				q_concat = q
				q_concat_len = q_len + 1
			else:
				q_concat, q_concat_len = concat_q(q_concat, q_concat_len, qbot_decoding,
				                                  qbot_decoding_length, config.end_token)

			# Store qbot states
			c1_q, h1_q = qbot_states[0].c, qbot_states[0].h
			c2_q, h2_q = qbot_states[1].c, qbot_states[1].h

			# Obtain answer
			abot_feed = abot.make_true_decode_feed(qa, qa_len, q, q_len, image, c1_a, h1_a, c2_a,
			                                       h2_a)
			abot_fetches = [abot.hist_enc_state, abot.decoder_pred_decode,
			                abot.decoder_outputs_length_decode]
			abot_states, abot_decoding, abot_decoding_length = abot_session.run(abot_fetches,
			                                                                    feed_dict=abot_feed)

			# Store abot states
			c1_a, h1_a = abot_states[0].c, abot_states[0].h
			c2_a, h2_a = abot_states[1].c, abot_states[1].h

			# Concatenate qa
			qa, qa_len = concatenate_qa(qbot_decoding, abot_decoding, qbot_decoding_length,
			                            abot_decoding_length, config.end_token)

		print('Finished dialogs for batch {}'.format(cur_batch))
		# Dialog finished, make prediction
		qbot_feed = qbot.make_true_decode_feed(qa, qa_len, c1_q, h1_q, c2_q, h2_q)
		prediction = qbot_session.run(qbot.y_t, feed_dict=qbot_feed)

		print('Obtained predictions for batch {}'.format(cur_batch))
		batch_ranks = ranks(prediction, val_images, batch['img_inds'])

		print('Obtained rankings for batch {}'.format(cur_batch))
		all_ranks.extend(batch_ranks)

	scores = list(range(eval_size))
	percentiles = list(
		map(lambda rank: stats.percentileofscore(scores, eval_size - rank), all_ranks))

	print('Mean percentile ranks is {}'.format(np.mean(percentiles)))
	print('Mean rank is {}'.format(np.mean(all_ranks)))


def rank_both_rounds(qbot, abot, data, qbot_session, abot_session, config, eval_size=10000):
	print('Started ranking...')
	assert eval_size <= 40000

	if config.dataset == '09':
		val_images = data.data['val_img_fv'][:eval_size]
		val_indices = data.data['val_img_pos'][:eval_size]
	else:
		val_images = data.data['test_img_fv'][:eval_size]
		val_indices = data.data['test_img_pos'][:eval_size]
	dataset = 'val' if config.dataset == '09' else 'test'
	num_rounds = 10
	batch_size = config.batch_size
	all_ranks = [[] for _ in range(10)]  # ranking of validation set at each round
	for cur_batch in range(0, eval_size, batch_size):
		if cur_batch % 1000 == 0:
			print('Ranking at batch, ', cur_batch)
		batch, *_ = data.get_test_batch(start_id=cur_batch, batch_size=batch_size, subset=dataset)

		c1_q = np.zeros((config.batch_size, config.history_dim))
		h1_q = np.zeros((config.batch_size, config.history_dim))
		c2_q = np.zeros((config.batch_size, config.history_dim))
		h2_q = np.zeros((config.batch_size, config.history_dim))

		c1_a = np.zeros((config.batch_size, config.history_dim))
		h1_a = np.zeros((config.batch_size, config.history_dim))
		c2_a = np.zeros((config.batch_size, config.history_dim))
		h2_a = np.zeros((config.batch_size, config.history_dim))

		qa = batch['history'][:, 0, :]
		qa_len = batch['history_lengths'][:, 0]
		image = batch['img_feats']

		# q_concat, q_concat_len = None, None
		for cur_round in range(num_rounds):
			# Obtain question and process it
			qbot_feed = qbot.make_true_decode_feed(qa, qa_len, c1_q, h1_q, c2_q, h2_q)
			qbot_fetches = [qbot.hist_enc_state, qbot.decoder_pred_decode,
			                qbot.decoder_outputs_length_decode]

			qbot_states, qbot_decoding, qbot_decoding_length = qbot_session.run(qbot_fetches,
			                                                                    feed_dict=qbot_feed)
			q, q_len = preprocess_q(qbot_decoding, qbot_decoding_length, config.end_token)

			# if cur_round == 0:
			# 	q_concat = q
			# 	q_concat_len = q_len + 1
			# else:
			# 	q_concat, q_concat_len = concat_q(q_concat, q_concat_len, qbot_decoding,
			# 	                                  qbot_decoding_length, config.end_token)

			# Store qbot states
			c1_q, h1_q = qbot_states[0].c, qbot_states[0].h
			c2_q, h2_q = qbot_states[1].c, qbot_states[1].h

			# Obtain answer
			abot_feed = abot.make_true_decode_feed(qa, qa_len, q, q_len, image, c1_a, h1_a, c2_a,
			                                       h2_a)
			abot_fetches = [abot.hist_enc_state, abot.decoder_pred_decode,
			                abot.decoder_outputs_length_decode]
			abot_states, abot_decoding, abot_decoding_length = abot_session.run(abot_fetches,
			                                                                    feed_dict=abot_feed)

			# Store abot states
			c1_a, h1_a = abot_states[0].c, abot_states[0].h
			c2_a, h2_a = abot_states[1].c, abot_states[1].h

			# Concatenate qa
			qa, qa_len = concatenate_qa(qbot_decoding, abot_decoding, qbot_decoding_length,
			                            abot_decoding_length, config.end_token)

			# Make predictions
			qbot_feed = qbot.make_true_decode_feed(qa, qa_len, c1_q, h1_q, c2_q, h2_q)

			prediction = qbot_session.run(qbot.y_t, feed_dict=qbot_feed)
			print('Obtained predictions at round {}, now ranking them..'.format(cur_round + 1))
			batch_ranks = ranks(prediction, val_images, batch['img_inds'])
			all_ranks[cur_round].extend(batch_ranks)
			print('Ranking at round {} finished'.format(cur_round + 1))

		print('Finished dialogs for batch {}'.format(cur_batch))

	for cur_round in range(num_rounds):
		scores = list(range(eval_size))
		percentiles = list(
			map(lambda rank: stats.percentileofscore(scores, eval_size - rank),
			    all_ranks[cur_round]))

		print('Round {}: Mean percentile ranks is {}'.format(cur_round + 1, np.mean(percentiles)))
		print('Round{}: Mean rank is {}'.format(cur_round + 1, np.mean(all_ranks[cur_round])))
		print('---' * 20)


def concat_q(concat, concat_length, q, q_len, eos_token):
	ques = np.copy(q[:, :, 0])
	new_concat = np.concatenate((concat, np.zeros(shape=ques.shape, dtype=np.int32)), axis=0)
	batch_size = new_concat.shape[1]
	lengths = np.zeros((batch_size,), dtype=np.int32)
	for i in range(batch_size):
		curr_q = ques[:, i]
		curr_q_len = np.argmax(curr_q == eos_token)
		con_len = concat_length[i]
		new_concat[con_len: con_len + curr_q_len + 1, i] = curr_q[:curr_q_len + 1]
		lengths[i] = con_len + curr_q_len + 1

	return new_concat, lengths


def preprocess_q(q, q_length, eos_token):
	# [max_time_steps, batch_size, beam_width] -> [max_time_steps, batch_size]
	pq = np.copy(q[:, :, 0])

	# [batch_size, beam_width] -> [batch_size,]
	pq_length = np.copy(q_length[:, 0])

	# Adjust lengths for each batch
	for i in range(pq.shape[1]):
		qb = pq[:, i]
		q_len = np.argmax(qb == eos_token)
		pq_length[i] = q_len

	return pq, pq_length


def concatenate_qa(q, a, q_length, a_length, eos_token):
	# 1. Select only the most probable beam decoding

	# [max_time_steps, batch_size, beam_width] -> [max_time_steps, batch_size]
	q = q[:, :, 0]
	a = a[:, :, 0]

	# [batch_size, beam_width] -> [batch_size,]
	q_length = q_length[:, 0]
	a_length = a_length[:, 0]

	# 2. Slice q,a to longest element length, along time steps axis
	q = q[:np.max(q_length), :]
	a = a[:np.max(a_length), :]

	# 3. Pad question with length of answer, to make room for concatenation
	qa = np.concatenate((q, np.zeros(shape=a.shape, dtype=np.int32)), axis=0)

	# 4. Insert array into questions
	batch_size = q.shape[1]
	lengths = np.zeros((batch_size,), dtype=np.int32)
	for i in range(qa.shape[1]):
		# Question and answer for batch i
		qb = qa[:, i]
		ab = a[:, i]

		# Handles the case where multiple eos tokens are printed
		# Note: make sure eos tokens are always printed at least once.
		q_len = np.argmax(qb == eos_token)
		a_len = np.argmax(ab == eos_token)

		# Insert answer into padded question array
		qa[q_len:q_len + a_len, i] = a[:a_len, i]

		# Register length
		lengths[i] = q_len + a_len

	# 5. Slice concatenation to longest element length.
	qa = qa[:np.max(lengths), :]

	return qa, lengths


parser = argparse.ArgumentParser()

# Dataset parameters

parser.add_argument('--save_path', type=str, default='new_trained_models/')
parser.add_argument('--logs_path', type=str, default='new_logs/')
parser.add_argument('--model_name', type=str, default='final')
parser.add_argument('--qbot_model_name', type=str, default='qbot')
parser.add_argument('--abot_model_name', type=str, default='abot')

# Network parameters
parser.add_argument('--embed_dim', type=int, default=300)
parser.add_argument('--fact_dim', type=int, default=512)
parser.add_argument('--history_dim', type=int, default=512)
parser.add_argument('--decoder_dim', type=int, default=512)
parser.add_argument('--cell_type', type=str, default='LSTM')
parser.add_argument('--num_layers', type=int, default=2, help='Number of nlayers in LSTM')

# Decoding params
parser.add_argument('--beam_width', type=int, default=6)
parser.add_argument('--max_decode_step', type=int, default=20)

# Data processing parameters
parser.add_argument('--img_norm', type=int, default=1)
parser.add_argument('--img_feature_size', type=int, default=4096)
parser.add_argument('--time_major', default=1, type=int, choices=[0, 1])
parser.add_argument('--max_history_len', type=int, default=60)

# Training parameters
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'decode', 'rank', 'test'])
parser.add_argument('--rank_rounds', type=int, default=0)
parser.add_argument('--bot', type=str, default='qbot', choices=['qbot', 'abot', 'both'])
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size (number of threads) (Adjust base on GPU memory)')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
parser.add_argument('--num_epochs', type=int, default=15, help='Epochs')
parser.add_argument('--lr_rate_decay', type=int, default=10,
                    help='After lr_decay epochs lr reduces to 0.1*lr')
parser.add_argument('--lr_decay_rate', type=float, default=0.9997592083,
                    help='Decay for learning rate')

parser.add_argument('--dataset', type=str, default='09', choices=['09', '05'])
parser.add_argument('--min_lr_rate', type=float, default=5e-5,
                    help='Minimum learning rate')

flags = parser.parse_args()
flags.rank_rounds = bool(flags.rank_rounds)
flags.input_img = 'data/data_img.h5' if flags.dataset == '09' else 'data/data_img_05.h5'
flags.input_data = 'data/visdial_data.h5' if flags.dataset == '09' else 'data/visdial_data_05.h5'
flags.input_json = 'data/visdial_params.json' if flags.dataset == '09' else 'data/visdial_params_05.json'


def fetch_dataloader():
	loader_file = 'data_loader.pkl' if flags.dataset == '09' else 'data_loader_05.pkl'
	if os.path.isfile(loader_file):
		data_loader = dill.load(open(loader_file, 'rb'))
	else:
		data_loader = dataloader.DataLoader(flags,
		                                    ['train', 'val'] if flags.dataset == '09' else ['train',
		                                                                                    'test'])
		dill.dump(data_loader, open(loader_file, 'wb'))
	return data_loader


def fetch_model(session, config):
	if config.mode in ['train', 'test']:
		if config.bot == 'qbot':
			print('Running qbot...')
			model = QBOT(session, config, config.mode)
		else:
			print('Running abot...')
			model = ABOT(session, config, config.mode)
	else:
		model = load_model(session, config, config.mode)

	return model


def load_model(session, config, mode='train', epoch=None):
	print('Reloading.. {}'.format(config.bot))
	if config.bot == 'qbot':
		model = QBOT(session, config, mode)
	else:
		model = ABOT(session, config, mode)

	if epoch is not None:
		save_path = os.path.join(flags.save_path, 'epoch_{}'.format(int(epoch)),
		                         'model.ckpt')
	else:
		save_path = os.path.join(flags.save_path, config.model_name, 'model.ckpt')

	print('Reloading model parameters from save path {}'.format(save_path))
	model.restore(session, save_path)
	return model


def training():
	data_loader = fetch_dataloader()
	config = flags
	# # TODO(Mircea): Should the start token be in the vocabulary?

	config.start_token = data_loader.data['word2ind']['<START>']
	config.end_token = data_loader.data['word2ind']['<EOS>']
	config.pad_token = 0
	config.vocab_dim = data_loader.vocab_size
	config.logs_path = os.path.join(config.logs_path, config.model_name)
	config.batch_size = 2 if config.mode == 'decode' else config.batch_size

	if tf.gfile.Exists(config.logs_path) and config.mode == 'train':
		tf.gfile.DeleteRecursively(config.logs_path)
	tf.gfile.MakeDirs(config.logs_path)

	with tf.Session(
			config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
		model = fetch_model(sess, config)
		if config.mode == 'train':
			model.train(data_loader, config.num_epochs)
		elif config.mode == 'decode':
			model.decode(data_loader)
		elif config.mode == 'rank':
			model.rank(data_loader)
		elif config.mode == 'test':
			model.test(data_loader)


def test_both_bots():
	data_loader = fetch_dataloader()
	config = flags

	print('Fetching type of ques lens: {}'.format(data_loader.data['test_ques_len'].dtype))

	if config.mode == 'train':
		raise Exception('Both bots can only be tested for decoding or ranking.')

	config.start_token = data_loader.data['word2ind']['<START>']
	config.end_token = data_loader.data['word2ind']['<EOS>']
	config.pad_token = 0
	config.vocab_dim = data_loader.vocab_size
	config.batch_size = 2 if config.mode == 'decode' else config.batch_size

	qbot_graph = tf.Graph()
	with qbot_graph.as_default():
		qbot_session = tf.Session(
			config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

		# Load qbot
		config.logs_path = os.path.join(config.logs_path, config.qbot_model_name)
		config.model_name = config.qbot_model_name
		config.bot = 'qbot'
		qbot = load_model(qbot_session, config, config.mode)

	abot_graph = tf.Graph()
	with abot_graph.as_default():
		abot_session = tf.Session(
			config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

		# Load abot
		config.logs_path = os.path.join(config.logs_path, config.abot_model_name)
		config.model_name = config.abot_model_name
		config.bot = 'abot'
		abot = load_model(abot_session, config, config.mode)

	if config.mode == 'decode':
		decode_both(qbot, abot, data_loader, qbot_session, abot_session, config)
	elif config.mode == 'rank':
		if config.rank_rounds:
			rank_both_rounds(qbot, abot, data_loader, qbot_session, abot_session, config,
			                 eval_size=40000 if config.dataset == '09' else 9500)
		else:
			rank_both(qbot, abot, data_loader, qbot_session, abot_session, config,
			          eval_size=40000 if config.dataset == '09' else 9500)


if __name__ == '__main__':
	if flags.bot == 'both':
		test_both_bots()
	else:
		training()
