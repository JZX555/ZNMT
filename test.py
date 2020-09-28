import torch
from model.Transformer import Transformer
from driver.Config import Configurable
from data.vocabulary import Vocabulary
from driver.Optim import Optimizer

a = torch.ones((5, 10), dtype=torch.bool)
b = torch.zeros((5, 10), dtype=torch.bool)
torch.gt(a + b, 0)

config = Configurable('./default.cfg', [])
print(torch.__version__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gpu = torch.cuda.is_available()
if gpu:
    config.use_cuda = True
    torch.cuda.set_device(0)
    print("GPU ID: ", 0)
print("\nGPU using status: ", config.use_cuda)

src_vocab = Vocabulary(config.src_vocab_type, config.src_vocab_path)
tgt_vocab = Vocabulary(config.tgt_vocab_type, config.tgt_vocab_path)


model = Transformer(config, src_vocab.max_n_words, tgt_vocab.max_n_words)
model.to(device)
optim = Optimizer(name=config.learning_algorithm,
                    model=model,
                    lr=config.learning_rate,
                    grad_clip=config.clip
                    )

a = torch.ones((5, 10), dtype=torch.int64).cuda()
b = torch.ones((5, 10), dtype=torch.int64).cuda()

c = model(a, b, lengths = 0)

print(c)
print(c.size())

# torch.save(model.encoder.embeddings.embeddings.state_dict(), './save/test.ckpt')

# state_dict = torch.load('./save/test.ckpt')

# print(state_dict)