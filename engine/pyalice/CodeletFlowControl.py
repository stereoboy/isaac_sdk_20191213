'''
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''


class CodeletFlowControl(object):
    """Python flow control primitive that helps Python codelet backend to execute the job that C++
    PyCodeletFlowControl sends

    Args:
        owner (CodeletBackend): the object that actually is able to execute the job
        delegator (PybindPyCodelet): the pybind C++ pycodelet that sends the job to python side

    Attributes:
        same as above
  """

    def __init__(self, owner, delegator):
        self.owner = owner
        self.delegator = delegator

    def run(self):
        """Execution primitive for performing a job that the C++ PyCodelet delegates. The run function
    will first request for a job and perform the job. Upon completing the job the function notifies
    the C++ PyCodelet.

    Returns:
        False if only when the codelet needs to be stopped
    """
        job = self.delegator.python_wait_for_job()    # request for a job
        # Null job means exiting
        if not job:
            return False
        assert hasattr(self.owner, job), \
            "python flow controller {} cannot execute job {} (attribute not found)".format(owner, job)
        job_callable = getattr(self.owner, job)
        assert callable(job_callable), \
            "python flow controller {} cannot execute job {} (not callable)".format(owner, job)
        job_callable()
        # notifies the delegator that the job is completed
        self.delegator.python_job_finished()
        return True
