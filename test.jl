using Base.Iterators: partition
using CuArrays, CUDAnative, CUDAdrv
using Flux
using Flux: onehotbatch, throttle
using Flux.Tracker: back!, update!
using Images
using MLDatasets
using Plots

device!(CuDevice(0))

println("load datasets")

fashion_x, fashion_y = FashionMNIST.traindata(Float64)
test_fashion_x, test_fashion_y = FashionMNIST.testdata(Float64)

resized(img) = imresize(img, (32,32))

fashion_x = real.(mapslices(resized, MNIST.convert2image(fashion_x), dims=[1,2]))
test_fashion_x = real.(mapslices(resized, MNIST.convert2image(test_fashion_x), dims=[1,2]))

fashion_x = reshape(fashion_x, (32,32,1,60000))
test_fashion_x = reshape(test_fashion_x, (32,32,1,10000))

mnist_y = onehotbatch(mnist_y, 0:9)
test_fashion_y = onehotbatch(test_fashion_y, 0:9)

fashion_train = [(cat(fashion_x[:,:,:,i], dims=4), fashion_y[:,i]) |> gpu
    for i in partition(1:60000, 200)]

loss(model, x, y) = Flux.mse(model(x), y)

model_creator() = Chain(
  Conv((3, 3), 1 => 64, relu, pad=(1, 1), stride=(1, 1)),
  #BatchNorm(64),
  x -> maxpool(x, (2, 2)),
  Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
  #BatchNorm(128),
  x -> maxpool(x, (2,2)),
  Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
  #BatchNorm(256),
  x -> maxpool(x, (2, 2)),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  x -> maxpool(x, (2, 2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(1024, 4096, relu),
  Dropout(0.5),
  Dense(4096, 4096, relu),
  Dropout(0.5),
  Dense(4096, 10),
  softmax) |> gpu

m = model_creator()

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

test_fashion = (test_fashion_x, test_fashion_y) |> gpu
accuracy(m, x, y) = mean(argmax(m(x).data, dims=1) .== argmax(y, dims=1))

l(x,y) = loss(m,x,y)

a(x,y) = accuracy(m,x,y)

opt = ADAM(params(m))
opt2 = ADAM(params(m2))

lp = LossPlot(plot([l(test_fashion...).data]), false)

Flux.train!(l, fashion_train, opt, cb = throttle(() -> lp(l(test_fashion...).data), 2))
save_plot(lp, "test")
