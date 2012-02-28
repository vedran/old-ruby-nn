class Neuron
  attr_accessor :out_weights, :out_values, :input

  def initialize
    @out_weights = Array.new
    @out_values = Array.new
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

    for i in (0...input_count)
      new_neuron = Neuron.new

      for j in (0...out_connections_count)
        new_neuron.out_weights[j] = ((2.0 * Random.rand.round(1)).to_f - 1)
      end

      new_layer.neurons << new_neuron
    end

    return new_layer
  end

  def self.new_output_layer(num_neurons)
    output_layer = Layer.new

    for i in (0...num_neurons)
      output_layer.neurons << Neuron.new
    end

    return output_layer
  end
end

class Neural_Network

  attr_reader :layers, :delta_weights

  def initialize(input_count, hidden_count, output_count)
    @output_error_values = Array.new
    @hidden_error_values = Array.new
    @delta_weights = Array.new
    @layers = Array.new

    @layers << Layer.new_internal_layer(input_count, hidden_count)

    bias_neuron = Neuron.new

    for k in (0...hidden_count)
      bias_neuron.out_weights[k] = 1.0
      bias_neuron.out_values[k] = 1.0
      bias_neuron.input = 1.0
    end

    @layers[0].neurons << bias_neuron

    @layers << Layer.new_internal_layer(hidden_count, output_count)
    @layers << Layer.new_output_layer(output_count)
  end

  def activation_function(x)
    result = Math.tanh(x).round(4)
  end

  def derived_activation_function(y)
    1 - (y * y)
  end

  def fire_neurons(input)
    input.each_with_index do |value, i|
      for j in (0...@layers[1].neurons.count) do
        @layers[0].neurons[i].input = value.to_f
        @layers[0].neurons[i].out_values[j] =
          activation_function(@layers[0].neurons[i].out_weights[j] * value.to_f)
      end
    end

    for i in (0...@layers.count)
      @layers[i].neurons.each_with_index do |neuron, j|
        cur_output = 0

        next_layer_count = i < @layers.length - 1 ? @layers[i+1].neurons.length : 1

        if i > 0 && i < @layers.length #hidden and output layer calculations
          while cur_output < next_layer_count
            sum = 0.0

            @layers[i-1].neurons.each do |prev_neuron|
              sum += prev_neuron.out_values[j] * prev_neuron.out_weights[j]
            end

            neuron.input = sum

            #activation calculation
            neuron.out_values[cur_output] = activation_function(sum).to_f
            cur_output += 1
          end
        end
      end      
    end

  end
  
  def calculate_errors(desired_output)
        #set output layer error values
    @layers[2].neurons.each_with_index do |neuron, j|
      @output_error_values[j] = (-(desired_output[j] - neuron.out_values[0]) *
                                 (1 - neuron.out_values[0]) * neuron.out_values[0])
    end
  end



  def learn(desired_output, learning_constant)
    w_index = 0
    input_layer  = @layers[0]
    hidden_layer = @layers[1]
    output_layer = @layers[2]

    hidden_layer.neurons.each_with_index do |neuron, j|
      for i in (0...output_layer.neurons.count)
        @hidden_error_values[j] = (-@output_error_values[i] * (1 - output_layer.neurons[i].out_values[0]) *
                                   output_layer.neurons[i].out_values[0])
        delta_weights[w_index] = (learning_constant * @output_error_values[i] * neuron.input)
        neuron.out_weights[i] += delta_weights[w_index]
        w_index += 1
      end
    end

    #change input-to-hidden weights
    input_layer.neurons.each do |neuron|
      for i in (0...hidden_layer.neurons.count)
        delta_weights[w_index] = (learning_constant * @hidden_error_values[i] * neuron.input)
        neuron.out_weights[i] += delta_weights[w_index]
        w_index += 1
      end
    end
  end
end

#2-bit adder
input = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
desired_output = [[0, 0], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1, 1]]
network = Neural_Network.new(3, 3, 2)

total_connections = network.layers[0].neurons.count * network.layers[1].neurons.count * 
  network.layers[2].neurons.count
for i in (1...100000)
  delta_weight_sums = Array.new(total_connections) { 0.0 }
  for j in (0...input.count)

    network.fire_neurons(input[j]);
    network.calculate_errors(desired_output[j])
    network.learn(desired_output[j], 0.05)

    network.delta_weights.each_with_index do |weight, w|
      delta_weight_sums[w] += weight.abs
    end

    puts "test case #{j} of epoch #{i} has a max delta_weight of #{delta_weight_sums.max}"
  end

  if delta_weight_sums.max <= 0.0005
    puts "learned successfully!"
    break
  end
end

puts "finished learning attempts"

puts
puts network.layers[0].inspect
puts
puts network.layers[1].inspect
puts
puts network.layers[2].inspect
puts
