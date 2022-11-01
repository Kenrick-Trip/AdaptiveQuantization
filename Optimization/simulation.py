import numpy as np
import matplotlib.pyplot as plt


# Based on https://github.com/williewheeler/stats-demos/blob/master/queueing/single-queue-sim.ipynb
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

        self.q_service_t = np.zeros([self.n])
        self.uq_service_t = np.zeros([self.n])
        self.cont_1_service_t = np.zeros([self.n])
        self.cont_2_service_t = np.zeros([self.n])
        self.cont_3_service_t = np.zeros([self.n])

        # specify arrays:
        self.specify_data()

    def specify_data(self):
        # for MG1 system:
        setting = "load"
        sys = "test"

        if setting == "save" and sys == "test":
            inter_arrivals = np.random.exponential(scale=1/self.lambda_mean, size=self.n)
            self.arrival_t = np.cumsum(inter_arrivals)
            np.save("arrival_t.npy", self.arrival_t)
            self.q_service_t = np.abs(np.random.normal(loc=self.mean_s_q, scale=self.std_s_q, size=self.n))
            np.save("qservice_t.npy", self.q_service_t)
            uq_service_t = np.abs(np.random.normal(loc=self.mean_s_uq, scale=self.std_s_uq, size=self.n))
            np.save("uqservice_t.npy", uq_service_t)
            self.cont_1_service_t = uq_service_t
            self.cont_2_service_t = uq_service_t
            self.cont_3_service_t = uq_service_t
            self.uq_service_t = uq_service_t

        elif setting == "save" and sys == "real":
            inter_arrivals = np.random.exponential(scale=1/self.lambda_mean, size=self.n)
            self.arrival_t = np.cumsum(inter_arrivals)
            np.save("rarrival_t.npy", self.arrival_t)
            self.q_service_t = np.abs(np.random.normal(loc=self.mean_s_q, scale=self.std_s_q, size=self.n))
            np.save("rqservice_t.npy", self.q_service_t)
            uq_service_t = np.abs(np.random.normal(loc=self.mean_s_uq, scale=self.std_s_uq, size=self.n))
            np.save("ruqservice_t.npy", uq_service_t)
            self.cont_1_service_t = uq_service_t
            self.cont_2_service_t = uq_service_t
            self.cont_3_service_t = uq_service_t
            self.uq_service_t = uq_service_t

        elif setting == "load" and sys == "test":
            self.arrival_t = np.load("arrival_t.npy")
            self.q_service_t = np.load("qservice_t.npy")
            uq_service_t = np.load("uqservice_t.npy")
            self.cont_1_service_t = uq_service_t
            self.cont_2_service_t = uq_service_t
            self.cont_3_service_t = uq_service_t
            self.uq_service_t = uq_service_t

        elif setting == "load" and sys == "real":
            self.arrival_t = np.load("rarrival_t.npy")
            self.q_service_t = np.load("rqservice_t.npy")
            uq_service_t = np.load("ruqservice_t.npy")
            self.cont_1_service_t = uq_service_t
            self.cont_2_service_t = uq_service_t
            self.cont_3_service_t = uq_service_t
            self.uq_service_t = uq_service_t

    def reset_times(self, controller_setting, controller_type):
        self.start_t = np.zeros(self.n)
        self.depart_t = np.zeros(self.n)

        # initial conditions:
        self.start_t[0] = self.arrival_t[0]

        if controller_setting == 1:
            self.jobs_quantized = self.n
            self.depart_t[0] = self.arrival_t[0] + self.q_service_t[0]
            for i in range(1, self.n):
                self.start_t[i] = max(self.arrival_t[i], self.depart_t[i - 1])
                self.depart_t[i] = self.start_t[i] + self.q_service_t[i]

        elif controller_setting == 2:
            self.depart_t[0] = self.arrival_t[0] + self.uq_service_t[0] # always start with unquatized taks
            for i in range(1, self.n):
                self.start_t[i] = max(self.arrival_t[i], self.depart_t[i - 1])
                self.depart_t[i] = self.start_t[i] + self.uq_service_t[i]

        elif controller_setting == 3 and controller_type == 0:
            self.depart_t[0] = self.arrival_t[0] + self.cont_1_service_t[0]  # always start with unquatized taks
            for i in range(1, self.n):
                self.start_t[i] = max(self.arrival_t[i], self.depart_t[i - 1])
                self.depart_t[i] = self.start_t[i] + self.cont_1_service_t[i]

        elif controller_setting == 3 and controller_type == 1:
            self.depart_t[0] = self.arrival_t[0] + self.cont_2_service_t[0]  # always start with unquatized taks
            for i in range(1, self.n):
                self.start_t[i] = max(self.arrival_t[i], self.depart_t[i - 1])
                self.depart_t[i] = self.start_t[i] + self.cont_2_service_t[i]

        elif controller_setting == 3 and controller_type == 2:
            self.depart_t[0] = self.arrival_t[0] + self.cont_3_service_t[0]  # always start with unquatized taks
            for i in range(1, self.n):
                self.start_t[i] = max(self.arrival_t[i], self.depart_t[i - 1])
                self.depart_t[i] = self.start_t[i] + self.cont_3_service_t[i]

    def quantization_controller(self, i, queue, controller_setting, controller_number):
        # controller:
        error = self.target_queue_size - queue

        controllers = ["conditional", "probabilistic", "stochastic"]
        controller_type = controllers[controller_number]

        # p = self.mean_s_uq/(self.mean_s_uq + self.mean_s_q)

        if i - queue > 0:
            if controller_type == "conditional":
                if not error > 0:
                    self.cont_1_service_t[i - queue] = self.q_service_t[i - queue]
                    self.reset_times(controller_setting, controller_number)
                    self.jobs_quantized += 1

            if controller_type == "probabilistic":
                p = 0.85
                r = np.random.rand()

                if error > 0:
                    if not r < p:
                        self.cont_2_service_t[i - queue] = self.q_service_t[i - queue]
                        self.reset_times(controller_setting, controller_number)
                        self.jobs_quantized += 1
                elif r < p:
                    self.cont_2_service_t[i - queue] = self.q_service_t[i - queue]
                    self.reset_times(controller_setting, controller_number)
                    self.jobs_quantized += 1

            if controller_type == "stochastic":
                p0 = 0.3
                k = 0.01
                p = np.clip(p0 - error*k, 0, 1)
                r = np.random.rand()

                if r < p:
                    self.cont_3_service_t[i - queue] = self.q_service_t[i - queue]
                    self.reset_times(controller_setting, controller_number)
                    self.jobs_quantized += 1



    def job_count(self, controller_setting, controller_type):
        # specify arrays:
        jobs_in_sys = []
        jobs_in_q = []
        job_event_times = []

        # log values
        self.jobs_quantized = 0
        self.quantizations = []

        self.specify_data()
        self.reset_times(controller_setting, controller_type)

        jobs_in_sys.append(0)
        jobs_in_q.append(0)
        job_event_times.append(0)
        self.quantizations.append(self.jobs_quantized)

        jobs_started = 0
        jobs_arrived = 0
        jobs_departed = 0
        events = 0

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
                jobs_in_sys.append(jobs_in_sys[events-1] + 1)
                jobs_in_q.append(jobs_in_q[events-1] + 1)
                job_event_times.append(arrived_job_t)

                if controller_setting == 3:
                    self.quantization_controller(jobs_arrived, jobs_in_q[-1], controller_setting, controller_type)
                    self.quantizations.append(self.jobs_quantized)

                jobs_arrived += 1

            elif started_job_t <= arrived_job_t and started_job_t <= departed_job_t:
                # event: a job is started
                jobs_in_sys.append(jobs_in_sys[events - 1])
                jobs_in_q.append(jobs_in_q[events - 1] - 1)
                job_event_times.append(started_job_t)
                jobs_started += 1
            else:
                # event: a job is departed
                jobs_in_sys.append(jobs_in_sys[events - 1] - 1)
                jobs_in_q.append(jobs_in_q[events - 1])
                job_event_times.append(departed_job_t)
                jobs_departed += 1

        mean_accuracy = ((self.n - self.jobs_quantized)*self.uq_accuracy + self.jobs_quantized*self.q_accuracy)/self.n

        print(self.jobs_quantized)

        return jobs_in_q, jobs_in_sys, job_event_times, mean_accuracy, self.quantizations, \
               np.linspace(0, self.n, self.n+1)

def plot_jobs_in_queue(sim):
    # sim.job_count(1) -> do only quantized jobs
    # sim.job_count(2) -> do only unquantized jobs
    # sim.job_count(3) -> apply the adaptive quantization controller

    aq2_queue, _, t, aq2_acc, quants2, j = sim.job_count(3, 2)
    q_queue, _, t, q_acc, _, _ = sim.job_count(1, 0)
    aq0_queue, _, t, aq0_acc, quants0, j = sim.job_count(3, 0)
    aq1_queue, _, t, aq1_acc, quants1, j = sim.job_count(3, 1)
    uq_queue, _, t, uq_acc, _, _ = sim.job_count(2, 0)

    print("Average accuracy with all jobs on quantized model: {} %".format(q_acc))
    print("Average accuracy with all jobs on unquantized model: {} %".format(uq_acc))
    print("Average accuracy with using the adaptive quantization controller: {} %".format(aq0_acc))
    print("Average accuracy with using the adaptive quantization controller: {} %".format(aq1_acc))
    print("Average accuracy with using the adaptive quantization controller: {} %".format(aq2_acc))

    plt.plot(t, q_queue, alpha=0.9, label="Quantized model, accuracy = {:.1f}%".format(q_acc), color="b")
    plt.plot(t, np.ones(len(t))*np.mean(q_queue), label="Mean queue size: {:.1f}".format(np.mean(q_queue)),
             linestyle="--", color="b")
    plt.plot(t, uq_queue, alpha=0.9, label="Unquantized model, accuracy = {:.1f}%".format(uq_acc), color="r")
    plt.plot(t, np.ones(len(t)) * np.mean(uq_queue), label="Mean queue size: {:.1f}".format(np.mean(uq_queue)),
             linestyle="--", color="r")
    plt.plot(t, aq0_queue, alpha=0.9, label="Conditional controller, accuracy = {:.1f}%".format(aq0_acc), color="g")
    plt.plot(t, np.ones(len(t)) * np.mean(aq0_queue), label="Mean queue size: {:.1f}".format(np.mean(aq0_queue)),
             linestyle="--", color="g")
    plt.plot(t, aq1_queue, alpha=0.9, label="Probabilistic controller, accuracy = {:.1f}%".format(aq1_acc), color="purple")
    plt.plot(t, np.ones(len(t)) * np.mean(aq1_queue), label="Mean queue size: {:.1f}".format(np.mean(aq1_queue)),
             linestyle="--", color="purple")
    plt.plot(t, aq2_queue, alpha=0.9, label="Stochastic controller, accuracy = {:.1f}%".format(aq2_acc), color="c")
    plt.plot(t, np.ones(len(t)) * np.mean(aq2_queue), label="Mean queue size: {:.1f}".format(np.mean(aq2_queue)),
            linestyle="--", color="c")
    plt.plot(t, np.ones(len(t)) * target_queue_size, label="Target queue size: {:.1f}".format(target_queue_size),
             linestyle="--", color="black")
    plt.legend(loc="upper left")
    plt.ylabel("Jobs in the queue")
    plt.xlabel("Time (s)")
    plt.title("Number of jobs in the queue over time")
    plt.show()

    plt.plot(j, quants0, alpha=0.9, label="Conditional controller", color="g")
    plt.plot(j, quants1, alpha=0.9, label="Probabilistic controller", color="purple")
    plt.plot(j, quants2, alpha=0.9, label="Stochastic controller", color="c")
    plt.legend(loc="upper left")
    plt.ylabel("Jobs quantized")
    plt.xlabel("jobs arrived")
    plt.title("The number of inference jobs on the quantized model")
    plt.show()




if __name__ == "__main__":
    setting = "test"
    if setting == "test":
        # system properties:
        num_tasks = 10000
        target_queue_size = 30

        # data from ANOVA:
        lambda_mean = 0.5

        # quantized job -> faster, less accurate
        # s equals service time for stability: 1/lambda_mean > mean_s
        mean_s_q = 1.8
        std_s_q = 0.2
        q_accuracy = 67

        # unquantized job -> slower, more accurate
        mean_s_uq = 1.998
        std_s_uq = 0.1
        uq_accuracy = 72
    elif setting == "real":
        # system properties:
        num_tasks = 10000
        target_queue_size = 30

        # data from ANOVA:
        lambda_mean = 23.37

        # quantized job -> faster, less accurate
        # s equals service time for stability: 1/lambda_mean > mean_s
        mean_s_q = 10.718951225280762 / 1000
        std_s_q = 0.01*mean_s_q
        q_accuracy = 99.25

        # unquantized job -> slower, more accurate
        mean_s_uq = 42.75380325317383 / 1000
        std_s_uq = 0.01*mean_s_uq
        uq_accuracy = 99.26



    sim = SimulateSystem(num_tasks, target_queue_size, lambda_mean, mean_s_q, std_s_q,
                 mean_s_uq, std_s_uq, q_accuracy, uq_accuracy)

    plot_jobs_in_queue(sim)

