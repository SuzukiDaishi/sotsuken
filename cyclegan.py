import util
import models
import dataloader
import datetime
import numpy as np
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
import warnings
import random
warnings.filterwarnings('ignore')

class CycleGAN(object):
    def __init__(self):
        # Input Shape
        self.hight = 36
        self.width = 128
        self.shape = (self.hight, self.width)

        # Datasets
        self.loader = dataloader.Dataloader('cache36_suzuki.pkl', 'cache36_kinoshita.pkl', shape=self.shape)
        self.loader.loadData()

        # Loss weights
        self.lambda_cycle = 10.  # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle  # Identity loss

        generator_optimizer = Adam(lr=0.0002, beta_1=0.5)
        discriminator_optimizer = Adam(lr=0.0001, beta_1=0.5)

        # Build and compile the discriminators
        self.d_A = models.build_PatchGAN_Discriminator(self.shape)
        self.d_B = models.build_PatchGAN_Discriminator(self.shape)
        self.d_A.compile(loss='mse', optimizer=discriminator_optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=discriminator_optimizer, metrics=['accuracy'])
        self.d_A.trainable = False
        self.d_B.trainable = False


        # Build and compile the generators
        self.g_AB = models.build_212CNN_Generator(self.shape)
        self.g_BA = models.build_212CNN_Generator(self.shape)

        input_A = Input(shape=self.shape)
        input_B = Input(shape=self.shape)

        fake_B = self.g_AB(input_A)
        fake_A = self.g_AB(input_B)

        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)


        id_A = self.g_BA(input_A)
        id_B = self.g_AB(input_B)

        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = Model(inputs=[input_A, input_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, id_A, id_B])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id],
                              optimizer=generator_optimizer)
    
    def train(self, epochs, sample_interval=50):
        start_time = datetime.datetime.now()
        valid = np.ones((1, 5, 16, 1))
        fake = np.zeros((1, 5, 16, 1))
        for epoch in range(epochs):
            #if epoch == 2:
            #    self.combined.loss_weights = [1, 1, self.lambda_cycle, self.lambda_cycle, 0, 0]
            for batch_i, (data_A, data_B) in enumerate(self.loader.loadBatch()):
                fake_B = self.g_AB.predict(data_A).reshape((1,)+self.shape)
                fake_A = self.g_BA.predict(data_B).reshape((1,)+self.shape)
                dA_loss_real = self.d_A.train_on_batch(data_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
                dB_loss_real = self.d_B.train_on_batch(data_B.reshape((1,)+self.shape), valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B.reshape((1,)+self.shape), fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                d_loss = 0.5 * np.add(dA_loss, dB_loss)
                g_loss = self.combined.train_on_batch([data_A, data_B], [valid, valid, data_A, data_B, data_A, data_B])
                elapsed_time = datetime.datetime.now() - start_time
                print("[Epoch %d/%d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s "
                      % (epoch, epochs, batch_i, d_loss[0], 100*d_loss[1], g_loss[0], np.mean(g_loss[1:3]),
                         np.mean(g_loss[3:5]), np.mean(g_loss[5:6]), elapsed_time))
                if batch_i % sample_interval == 0:
                    print("ここでセーブ")
                    self.sample_save(epoch, batch_i)

    def sample_save(self, epoch, batch_i):
        _, coded_sps_mean_A, coded_sps_std_A, coded_sps_max_A, _, _ = util.loadPickle('./cache36_suzuki.pkl')
        wave = util.loadWave(f'./datasets/suzuki/a01.wav')
        pwav = util.wavePadding(wave)
        f0, sp, ap = util.worldDecompose(pwav)
        coded_sp = util.worldEncodeSpectralEnvelop(sp)
        coded_sp_t = coded_sp.T
        coded_sp_norm = (coded_sp_t - coded_sps_mean_A) / coded_sps_max_A
        coded_sp_norm = coded_sp_norm[:,:128*6]
        coded_sp_norm = coded_sp_norm.reshape(6, 36, 128)
        dist = self.g_AB.predict(coded_sp_norm)
        dist = dist.reshape((36, 128*6))
        util.savePickle(f'./predict/log_a01_{epoch}_{batch_i}.pkl', dist)

if __name__ == "__main__":
    gan = CycleGAN()
    gan.train(25)
    gan.g_AB.save(f'./predict/log_cyclegan_AtoB_25.h5', include_optimizer=False)
    gan.g_BA.save(f'./predict/log_cyclegan_BtoA_25.h5', include_optimizer=False)
    gan.train(25)
    gan.g_AB.save(f'./predict/log_cyclegan_AtoB_50.h5', include_optimizer=False)
    gan.g_BA.save(f'./predict/log_cyclegan_BtoA_50.h5', include_optimizer=False)
    gan.train(25)
    gan.g_AB.save(f'./predict/log_cyclegan_AtoB_75.h5', include_optimizer=False)
    gan.g_BA.save(f'./predict/log_cyclegan_BtoA_75.h5', include_optimizer=False)
    gan.combined.loss_weights = [1, 1, gan.lambda_cycle, gan.lambda_cycle, 0, 0]
    gan.train(25)
    gan.g_AB.save(f'./predict/log_cyclegan_AtoB_100.h5', include_optimizer=False)
    gan.g_BA.save(f'./predict/log_cyclegan_BtoA_100.h5', include_optimizer=False)
