require 'csvigo'

mydata = csvigo.load("/home/abhishek/Desktop/sensor_time_data.csv")

in1 = torch.Tensor(mydata.TAG_2378)
in2 = torch.Tensor(mydata.TAG_2380)
in3 = torch.Tensor(mydata.TAG_2416)
in4 = torch.Tensor(mydata.TAG_2418)

input = torch.Tensor((#in1)[1],24)
input[{{},1}] = in1
input[{{},2}] = in2
input[{{},3}] = in3
input[{{},4}] = in4



output = torch.Tensor(mydata.TAG_2734)



index_all = torch.randperm(input.size(1))
index_train = 
index_test = 

train_X = input[{{1,train_boundary_index},{}}]
train_Y = output[{1,train_boundary_index}]

test_X = input[{{train_boundary_index,},{}}]
test_Y = output[{{train_boundary_index,},{}}]


mean = {}
std = {}
for i=1,input:size(2) do
  mean[i] = input[{{},i}]:mean()
  std[i] = input[{{},i}]:std()
  input[{{},i}]:add(-mean[i])
  input[{{},i}]:div(std[i])
end

local i = input:size(2)+1
mean[i] = output:mean()
std[i] = output:std()
output:add(-mean[i])
output:div(std[i])

dataset = {}
for i=1,input:size(1) do
  dataset[i] = {input[i], input[i]}
end

dataset.size = function(self)
  return input:size(1)
end

require 'nn'

mlp = nn.Sequential();  -- make a multi-layer perceptron
input_unit_count =  (#input)[2];

mlp:add(nn.Linear(input_unit_count, input_unit_count/2))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(input_unit_count/2, input_unit_count/4))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(input_unit_count/4, input_unit_count/2))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(input_unit_count/2, input_unit_count))

mlp:get(7).weight = mlp:get(1).weight:t()
mlp:get(7).gradWeight = mlp:get(1).gradWeight:t()
mlp:get(5).weight = mlp:get(3).weight:t()
mlp:get(5).gradWeight = mlp:get(3).gradWeight:t()


criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)

dataset = {}
for i=1,input:size(1) do
  out = torch.Tensor(1)
  out[1] = output[i]
  dataset[i] = {input[i], out}
end

dataset.size = function(self)
  return input:size(1)
end

mlp_new = nn.Sequential();  -- make a multi-layer perceptron
input_unit_count =  (#input)[2];

mlp_new:add(nn.Linear(input_unit_count, input_unit_count/2))
mlp_new:add(nn.Tanh())
mlp_new:add(nn.Linear(input_unit_count/2, input_unit_count/4))
mlp_new:add(nn.Tanh())
mlp_new:add(nn.Linear(input_unit_count/4, 1))

mlp_new:get(3).weight:copy(mlp:get(3).weight)
mlp_new:get(3).gradWeight:copy(mlp:get(3).gradWeight)

mlp_new:get(1).weight:copy(mlp:get(1).weight)
mlp_new:get(1).gradWeight:copy(mlp:get(1).gradWeight)

criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(mlp_new, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)
