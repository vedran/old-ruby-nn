require 'oily_png'


def parse_image(filename)
  image = ChunkyPNG::Image.from_file(filename)

  width = image.width
  height = image.height

  output = []

  width.times do |x|
    height.times do |y|
      output << ((ChunkyPNG::Color.r(image[x, y]) > 0 || ChunkyPNG::Color.g(image[x, y]) > 0 || ChunkyPNG::Color.b(image[x, y]) > 0) ? 1 : 0)
    end
  end

  output
end


class Neuron
  attr_accessor :out_weights, :output, :input

  def initialize
    @out_weights = Array.new
    @output = 0.0
    @input = 0.0
  end
end

class Layer
  attr_reader :neurons 

  def initialize
    @neurons = Array.new
  end

  def self.new_internal_layer(input_count, out_connections_count)
    new_layer = Layer.new

    input_count.times do |i|
      new_neuron = Neuron.new

      out_connections_count.times do |j|
        new_neuron.out_weights[j] = ((2.0 * Random.rand.round(1)).to_f - 1)
      end

      new_layer.neurons << new_neuron
    end

    return new_layer
  end

  def self.new_output_layer(num_neurons)
    output_layer = Layer.new

    num_neurons.times do |i|
      output_layer.neurons << Neuron.new
    end

    return output_layer
  end
end

class NeuralNetwork

  attr_reader :layers, :input_layer, :hidden_layer, :output_layer, :output_error_values

  def initialize(input_count, hidden_count, output_count)
    @output_error_values = Array.new
    @hidden_error_values = Array.new
    @layers = Array.new

    @layers << Layer.new_internal_layer(input_count, hidden_count)

    bias_neuron = Neuron.new

    hidden_count.times do |k|
      bias_neuron.out_weights[k] = 1.0
      bias_neuron.output = 1.0
      bias_neuron.input = 1.0
    end

    #@layers[0].neurons << bias_neuron

    @layers << Layer.new_internal_layer(hidden_count, output_count)
    @layers << Layer.new_output_layer(output_count)

    @input_layer = layers[0]
    @hidden_layer = layers[1]
    @output_layer = layers[2]
  end

  def activation_function(x)
    result = 1 / (1 + Math.exp(-x))
  end

  def fire_neurons(input)
    #input layer
    input.each_with_index do |value, i|
      @input_layer.neurons[i].input = @input_layer.neurons[i].output = value
    end

    #hidden layer
    @hidden_layer.neurons.each_with_index do |neuron, j|
      sum = 0.0

      @input_layer.neurons.each do |prev_neuron|
        sum += prev_neuron.output * prev_neuron.out_weights[j]
      end

      neuron.input = sum
      neuron.output = activation_function(sum).to_f
    end

    #output layer
    @output_layer.neurons.each_with_index do |neuron, j|
      sum = 0.0

      @hidden_layer.neurons.each do |prev_neuron|
        sum += prev_neuron.output * prev_neuron.out_weights[j]
      end

      neuron.input = sum
      neuron.output = activation_function(sum).to_f
    end
    @output_layer.neurons.collect {|n| n.output}
  end

  def learn(desired_output, learning_constant)
    #set output layer error values
    @output_layer.neurons.each_with_index do |neuron, j|
      @output_error_values[j] = neuron.output * (1 - neuron.output) * 
        (desired_output[j] - neuron.output)
    end

    @hidden_layer.neurons.each_with_index do |neuron, j|
      error_multiplier = 0.0
      total_output = 0.0
      @output_layer.neurons.count.times do |i|
        error_multiplier += neuron.out_weights[i] * @output_error_values[i]
        neuron.out_weights[i] += learning_constant * @output_error_values[i] * neuron.output
      end
      @hidden_error_values[j] = neuron.output * (1 - neuron.output) * error_multiplier
    end

    #change input-to-hidden weights
    @input_layer.neurons.each do |neuron|
      @hidden_layer.neurons.count.times do |i|
        neuron.out_weights[i] += learning_constant * @hidden_error_values[i] * neuron.input
      end
    end
  end
end

#2-bit adder

input = []
desired_output = []

#add circles to test data
8.times do |i|
  puts i
  input << parse_image('data/circle-' + i.to_s + '.png')
  input << parse_image('data/square-' + i.to_s + '.png')
  desired_output << [1, 0, 0]
  desired_output << [0, 1, 0]
end

puts "load weights? (y/n): "
choice = gets.chomp
if(choice == "y")
  if File.exists?("weights")
    weights_file = File.open("weights", "r")
    network = Marshal.load(File.binread("weights"))
  else 
    puts "Unable to load weights file!"
  end
end

#if loading the file failed and the network is nil, create a blank one
network ||= NeuralNetwork.new(256, 256, 3)

total_connections = network.input_layer.neurons.count * network.hidden_layer.neurons.count * 
  network.output_layer.neurons.count

max_error = 0.0
learning_constant = 0.005

1000000.times do |i|
  last_error = max_error
  max_error = 0.0

  input.count.times do |j|
    network.fire_neurons(input[j])
    network.learn(desired_output[j], learning_constant)#`0.055)

    desired_output[j].count.times do |k|
      error = (network.output_layer.neurons[k].output - desired_output[j][k]).abs
      if error > max_error
        max_error = error
      end
    end
  end

  puts "epoch " + i.to_s
  puts "Max Error: " + max_error.to_s
  

  if max_error < 5e-7 
    puts "learned successfully!"
    break
  end
end

puts "finished learning attempts"

gets

puts "Tests:"

input.count.times do |j|
  puts "Expected: " + desired_output.inspect + ", Got: " + network.fire_neurons(input[j]).inspect
  puts "Neurons: " + network.output_layer.inspect
end

puts "Override saved weights data?(y/n):" 
choice = gets.chomp

if(choice == "y")
  weights_file = File.new("weights", "wb")
  weights_file.puts Marshal.dump(network)
end
