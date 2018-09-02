using Base.Iterators: partition
using CuArrays
using Flux
using Flux: onehotbatch, throttle
using Flux.Tracker: back!, update!
using Images
using MLDatasets
using Plots

println("load datasets")

fashion_x, fashion_y = FashionMNIST.traindata(Float64)
cifar_x, cifar_y = CIFAR10.traindata(Float64)
mnist_x, mnist_y = MNIST.traindata(Float64)

resized(img) = imresize(img, (32,32))

mnist_x = real.(mapslices(resized, MNIST.convert2image(mnist_x), dims=[1,2]))
fashion_x = real.(mapslices(resized, MNIST.convert2image(fashion_x), dims=[1,2]))

fashion_x = reshape(fashion_x, (32,32,1,60000))
mnist_x = reshape(mnist_x, (32,32,1,60000))
cifar_x = reshape(real.(Gray.(CIFAR10.convert2image(cifar_x))), (32,32,1,50000))

mnist_y = onehotbatch(mnist_y, 0:9)
fashion_y = onehotbatch(fashion_y, 0:9)
cifar_y = onehotbatch(cifar_y, 0:9)

mnist_train = [(cat(mnist_x[:,:,:,i], dims=4), mnist_y[:,i])
    for i in partition(1:60000, 50)]
fashion_train = [(cat(fashion_x[:,:,:,i], dims=4), fashion_y[:,i])
    for i in partition(1:60000, 50)]
cifar_train = [(cat(cifar_x[:,:,:,i], dims=4), cifar_y[:,i])
    for i in partition(1:50000, 50)]

mnist_train = mnist_train |> cu
fashion_train = fashion_train |> cu
cifar_train = cifar_train |> cu

println("done datasets")

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
  softmax) |> cu

m2 = deepcopy(m)

loss(m, x, y) = Flux.mse(m(x), y)

for i in 1:40
    temp_model = deepcopy(m)
    opt = SGD(params(temp_model))
    task = rand([mnist_train, cifar_train])
    for j in 1:50
        x, y = task[rand(UInt64) % length(task) + 1]
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

lp = LossPlot(plot([loss(m, rand(fashion_train)...).data]), false)
lp2 = LossPlot(plot([loss(m2, rand(fashion_train)...).data]), false)

l(x,y) = loss(m,x,y)
l2(x,y) = loss(m2,x,y)

opt = ADAM(params(m))
opt2 = ADAM(params(m2))

Flux.train!(l, fashion_train, opt, cb = throttle(() -> lp(l(rand(fashion_train)...).data), 5))
Flux.train!(l2, fashion_train, opt2, cb = throttle(() -> lp2(l2(rand(fashion_train)...).data), 5))
save_plot(lp, "meta")
save_plot(lp2, "normal")
