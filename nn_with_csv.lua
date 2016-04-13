require 'csvigo'
require 'nn'

mydata = csvigo.load("/home/abhishek/Desktop/sensor_time_data.csv")

in1 = torch.Tensor(mydata.TAG_2378)
in2 = torch.Tensor(mydata.TAG_2445)
input = torch.Tensor((#in1)[1],2)
input[{{},1}] = in1
input[{{},2}] = in2
output = torch.Tensor(mydata.TAG_2657)

dataset = {}
for i=1,input:size(1) do
  out = torch.Tensor(1)
  out[1] = output[i]
  dataset[i] = {input[i], out}
end

dataset.size = function(self)
  return input:size(1)
end

mlp = nn.Sequential();  -- make a multi-layer perceptron
input_unit_count =  (#input)[2];
outputs_unit_count = 1; HUs = 20;

mlp:add(nn.Linear(input_unit_count, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs_unit_count))


criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)
