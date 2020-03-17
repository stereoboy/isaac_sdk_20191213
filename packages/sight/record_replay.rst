.. _rec-replay:

Record and Replay
=================================

An Isaac application is represented by a graph where the components inside their respective nodes
can receive and send messages. *Recording* is storing all the messages emitted by certain components
in a log. In the same way *Replay* means to replay all the recorded messages from a log.

The Isaac SDK provides two special components to achieve this purpose: **Recorder** and **Replay**.
Typically, in an app, two nodes are created that contain recorder and replay components
respectively, and these components are connected to other components depending upon the usecase.

In addition, Sight web server can be used to control either recording or replay from the websight
front end.

.. toctree::
   :hidden:

   doc/recordReplaySetup
   doc/recordReplaySight

