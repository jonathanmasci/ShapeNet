# http://learning-0mq-with-pyzmq.readthedocs.org/en/latest/pyzmq/devices/streamer.html
import sys
import zmq
import argparse

def main(N, inport, outport):
  try:
    context = zmq.Context(1)
    # Socket facing clients
    frontend = context.socket(zmq.PULL)
    frontend.sndhwm = N    
    frontend.rcvhwm = N
    #frontend.bind("tcp://127.0.0.1:5579")
    print "IN PORT: %s" % inport
    frontend.bind("tcp://127.0.0.1:" + inport)
    
    # Socket facing services
    backend = context.socket(zmq.PUSH)
    backend.sndhwm = N
    backend.rcvhwm = N
    #backend.bind("tcp://127.0.0.1:5580")
    print "OUT PORT: %s" % outport
    backend.bind("tcp://127.0.0.1:" + outport)

    zmq.device(zmq.STREAMER, frontend, backend)
  except Exception, e:
    print e
    print "bringing down zmq device"
  finally:
    backend.close()
    frontend.close()
    context.term()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Queue process.')
  parser.add_argument('--queue_size', metavar='queue_size', type=int,
          dest='queue_size',
          help='Maximum size of the queue',
          required=True)
  parser.add_argument('--inport', metavar='inport', type=str,
          dest='inport',
          help='Port to collect data from the generator',
          required=False, default='5579')
  parser.add_argument('--outport', metavar='outport', type=str,
          dest='outport',
          help='Port to send the data to the trainer',
          required=False, default='5580')
  
  args = parser.parse_args()

  main(args.queue_size, args.inport, args.outport)

