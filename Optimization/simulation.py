import numpy as np
import matplotlib.pyplot as plt


# used: https://github.com/williewheeler/stats-demos/blob/master/queueing/single-queue-sim.ipynb for inspiration
class SimulateSystem:
    def __init__(self, num_tasks, target_queue_size, lambda_mean, mean_s_q, std_s_q,
                 mean_s_uq, std_s_uq, q_accuracy, uq_accuracy):
        # job specifications: q = quantized, uq = unquantized
        self.lambda_mean = lambda_mean
        self.mean_s_q = mean_s_q
        self.mean_s_uq = mean_s_uq
        self.std_s_q = std_s_q
        self.std_s_uq = std_s_uq
        self.q_accuracy = q_accuracy
        self.uq_accuracy = uq_accuracy

        # system settings:
        self.n = num_tasks
        self.target_queue_size = target_queue_size

        # specify arrays:
        self.specify_data()

    def specify_data(self):
        # for MG1 system:
        inter_arrivals = np.random.exponential(scale=1/self.lambda_mean, size=self.n)
        self.arrival_t = np.cumsum(inter_arrivals)
        self.q_service_t = np.abs(np.random.normal(loc=self.mean_s_q, scale=self.std_s_q, size=self.n))
        self.uq_service_t = np.abs(np.random.normal(loc=self.mean_s_uq, scale=self.std_s_uq, size=self.n))

    def quantization_controller(self, i):
        # controller:
        error = self.jobs_in_q[-1] - self.target_queue_size

        if i < self.n:
            if error > 10:
                self.start_t[i] = max(self.arrival_t[i], self.depart_t[i - 1])
                self.depart_t[i] = self.start_t[i] + self.q_service_t[i]
                self.jobs_quantized += 1
            else:
                self.start_t[i] = max(self.arrival_t[i], self.depart_t[i - 1])
                self.depart_t[i] = self.start_t[i] + self.uq_service_t[i]

        # p0 = self.mean_s_q/(self.mean_s_uq + self.mean_s_q)
        # apply scaled sigmoid
        # k1 = 0.1
        # k2 = np.log((1-p0)/p0)/k1
        # p_control = 1/(1+np.e**(-k1*(error-k2)))

        #if i < self.n:
            #if self.p[i] < p_control:
               # self.start_t[i] = max(self.arrival_t[i], self.depart_t[i - 1])
              #  self.depart_t[i] = self.start_t[i] + self.q_service_t[i]
             #   self.jobs_quantized += 1
            #else:
                #self.start_t[i] = max(self.arrival_t[i], self.depart_t[i - 1])
                #self.depart_t[i] = self.start_t[i] + self.uq_service_t[i]


    def job_count(self, controller_setting):
        # specify arrays:
        self.start_t = np.zeros(self.n)
        self.depart_t = np.zeros(self.n)
        self.jobs_in_sys = []
        self.jobs_in_q = []
        self.job_event_times = []

        # log values
        self.jobs_quantized = 0

        # initial conditions:
        self.p = np.random.uniform(0, 1, self.n)
        self.start_t[0] = self.arrival_t[0]
        self.jobs_in_sys.append(0)
        self.jobs_in_q.append(0)
        self.job_event_times.append(0)
        jobs_started = 0
        jobs_arrived = 0
        jobs_departed = 0
        events = 0

        # initial schedule, using quantized or unquantized models
        if controller_setting == 1:
            self.jobs_quantized = self.n
            self.depart_t[0] = self.arrival_t[0] + self.q_service_t[0]
            for i in range(1, self.n):
                self.start_t[i] = max(self.arrival_t[i], self.depart_t[i - 1])
                self.depart_t[i] = self.start_t[i] + self.q_service_t[i]
        else:
            self.depart_t[0] = self.arrival_t[0] + self.uq_service_t[0] # always start with unquatized taks
            for i in range(1, self.n):
                self.start_t[i] = max(self.arrival_t[i], self.depart_t[i - 1])
                self.depart_t[i] = self.start_t[i] + self.uq_service_t[i]

        while jobs_departed < self.n:
            events += 1

            if jobs_started < self.n:
                started_job_t = self.start_t[jobs_started]
            else:
                started_job_t = np.infty

            if jobs_arrived < self.n:
                arrived_job_t = self.arrival_t[jobs_arrived]
            else:
                arrived_job_t = np.infty

            departed_job_t = self.depart_t[jobs_departed]

            if arrived_job_t <= started_job_t and arrived_job_t <= departed_job_t:
                # event: a job arrived at the queue
                self.jobs_in_sys.append(self.jobs_in_sys[events-1] + 1)
                self.jobs_in_q.append(self.jobs_in_q[events-1] + 1)
                self.job_event_times.append(arrived_job_t)
                jobs_arrived += 1

                if controller_setting == 3:
                    self.quantization_controller(jobs_arrived)

            elif started_job_t <= arrived_job_t and started_job_t <= departed_job_t:
                # event: a job is started
                self.jobs_in_sys.append(self.jobs_in_sys[events - 1])
                self.jobs_in_q.append(self.jobs_in_q[events - 1] - 1)
                self.job_event_times.append(started_job_t)
                jobs_started += 1
            else:
                # event: a job is departed
                self.jobs_in_sys.append(self.jobs_in_sys[events - 1] - 1)
                self.jobs_in_q.append(self.jobs_in_q[events - 1])
                self.job_event_times.append(departed_job_t)
                jobs_departed += 1

        mean_accuracy = ((self.n - self.jobs_quantized)*self.uq_accuracy + self.jobs_quantized*self.q_accuracy)/self.n

        print(self.jobs_quantized)

        return self.jobs_in_q, self.jobs_in_sys, self.job_event_times, mean_accuracy

def plot_jobs_in_queue(sim):
    # sim.job_count(1) -> do only quantized jobs
    # sim.job_count(2) -> do only unquantized jobs
    # sim.job_count(3) -> apply the adaptive quantization controller

    q_queue, _, t, q_acc = sim.job_count(1)
    uq_queue, _, t, uq_acc = sim.job_count(2)
    aq_queue, _, t, aq_acc = sim.job_count(3)

    print("Average accuracy with all jobs on quantized model: {} %".format(q_acc))
    print("Average accuracy with all jobs on unquantized model: {} %".format(uq_acc))
    print("Average accuracy with using the adaptive quantization controller: {} %".format(aq_acc))

    plt.plot(t, q_queue, alpha=0.9, label="Quantized model, accuracy = {:.2f}".format(q_acc), color="b")
    plt.plot(t, np.ones(len(t))*np.mean(q_queue), label="quantized mean: {:.2f}".format(np.mean(q_queue)),
             linestyle="--", color="b")
    plt.plot(t, uq_queue, alpha=0.9, label="Unquantized model, accuracy = {:.2f}".format(uq_acc), color="r")
    plt.plot(t, np.ones(len(t)) * np.mean(uq_queue), label="GG1 mean: {:.2f}".format(np.mean(uq_queue)),
             linestyle="--", color="r")
    plt.plot(t, aq_queue, alpha=0.9, label="Quantization controller, accuracy = {:.2f}".format(aq_acc), color="g")
    plt.plot(t, np.ones(len(t)) * np.mean(aq_queue), label="GG1 mean: {:.2f}".format(np.mean(aq_queue)),
             linestyle="--", color="g")
    plt.legend(loc="upper left")
    plt.ylabel("Jobs in the queue")
    plt.xlabel("Time (s)")
    plt.title("Number of jobs in the queue over time")
    plt.show()



if __name__ == "__main__":
    # system properties:
    num_tasks = 10000
    target_queue_size = 30

    # data from ANOVA:
    lambda_mean = 0.5

    # quantized job -> faster, less accurate
    # s equals service time for stability: 1/lambda_mean > mean_s
    mean_s_q = 1.8
    std_s_q = 0.2
    q_accuracy = 80

    # unquantized job -> slower, more accurate
    mean_s_uq = 1.995
    std_s_uq = 0.1
    uq_accuracy = 90


    sim = SimulateSystem(num_tasks, target_queue_size, lambda_mean, mean_s_q, std_s_q,
                 mean_s_uq, std_s_uq, q_accuracy, uq_accuracy)

    plot_jobs_in_queue(sim)

