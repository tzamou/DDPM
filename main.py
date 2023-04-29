from DDPM import DiffusionModel
from utils.config import TrainingConfig
import time



if __name__=='__main__':
    config = TrainingConfig(dataset_name="oxford_flowers102",
                            dataset_repetitions = 5,
                            num_epochs = 50,
                            image_size = 64,
                            kid_image_size = 75,
                            kid_diffusion_steps = 5,
                            plot_diffusion_steps = 20,
                            min_signal_rate = 0.02,
                            max_signal_rate = 0.95,
                            embedding_dims = 32,
                            embedding_max_frequency = 1000.0,
                            batch_size = 64,
                            ema = 0.999,
                            learning_rate = 1e-3,
                            weight_decay = 1e-4)

    t1 = time.time()
    model = DiffusionModel(config=config)
    model.set_callback()
    model.train()
    model.result()
    t2 = time.time()
    print('used time:',t2-t1)
