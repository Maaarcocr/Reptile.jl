using Flux
using MLDatasets
using CuArrays
using Flux: onehotbatch, throttle
using Flux.Tracker: back!, update!
using Images

println("load datasets")
fashion_x, fashion_y = FashionMNIST.traindata(Float64)
cifar_x, cifar_y = CIFAR10.traindata(Float64)
mnist_x, mnist_y = MNIST.traindata(Float64)

resized(img) = imresize(img, (32,32))

mnist_x = real.(mapslices(resized, MNIST.convert2image(mnist_x), dims=[1,2]))
fashion_x = real.(mapslices(resized, MNIST.convert2image(fashion_x), dims=[1,2]))

fashion_x = reshape(fashion_x, (32,32,1,60000)) |> gpu
mnist_x = reshape(mnist_x, (32,32,1,60000)) |> gpu
cifar_x = reshape(real.(Gray.(CIFAR10.convert2image(cifar_x))), (32,32,1,50000)) |> gpu

mnist_y = onehotbatch(mnist_y, 0:9) |> gpu
fashion_y = onehotbatch(fashion_y, 0:9) |> gpu
cifar_y = onehotbatch(cifar_y, 0:9) |> gpu

println("done datasets")

function get_sample(xs, ys)
    index = rand(UInt64) % (size(xs)[length(size(xs))] - 4) + 1
    (xs[:,:,:,index:index+4], ys[:, index:index+4])
end

get_task() = [(cifar_x, cifar_y), (mnist_x, mnist_y)][rand(UInt64) % 2 + 1]

m = Chain(
  Conv((3, 3), 1 => 64, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64),
  Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64),
  x -> maxpool(x, (2, 2)),
  Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128),
  Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128),
  x -> maxpool(x, (2,2)),
  Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  x -> maxpool(x, (2, 2)),
  Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  x -> maxpool(x, (2, 2)),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  x -> maxpool(x, (2, 2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(512, 4096, relu),
  Dropout(0.5),
  Dense(4096, 4096, relu),
  Dropout(0.5),
  Dense(4096, 10),
  softmax) |> gpu

m2 = deepcopy(m)

loss(m, x, y) = Flux.mse(m(x), y)

for i in 1:1000
    temp_model = deepcopy(m)
    opt = SGD(params(temp_model))
    task_x, task_y = get_task()
    for j in 1:50
        x, y = get_sample(task_x, task_y)
        l = loss(temp_model, x, y)
        back!(l)
        opt()
    end
    old_params = params(m)
    new_params = params(temp_model)
    for i in range(1,length=length(old_params))
        update!(old_params[i], (new_params[i] - old_params[i]) * 0.1)
    end
end

println("end meta learning")

using Plots

struct LossPlot
    plt::Plots.Plot
    display::Bool
end

function (lp::LossPlot)(loss)
    push!(lp.plt, [loss])
    if lp.display
        display(lp.plt)
    end
end

function save_plot(lp::LossPlot, name)
    png(lp.plt, name)
end

lp = LossPlot(plot([loss(m, get_sample(fashion_x,fashion_y)...).data]), false)
lp2 = LossPlot(plot([loss(m2, get_sample(fashion_x,fashion_y)...).data]), false)

l(x,y) = loss(m,x,y)
l2(x,y) = loss(m2,x,y)

opt = ADAM(params(m))
opt2 = ADAM(params(m2))

using Base.Iterators: partition
train = [(cat(fashion_x[:,:,:,i], dims=4), fashion_y[:,i])
    for i in partition(1:60_000, 100)] |> gpu

Flux.train!(l, train, opt, cb = throttle(() -> lp(l(get_sample(fashion_x, fashion_y)...).data), 5))
Flux.train!(l2, train, opt2, cb = throttle(() -> lp2(l2(fashion_x, fashion_y).data), 5))
save_plot(lp, "meta")
save_plot(lp2, "normal")
