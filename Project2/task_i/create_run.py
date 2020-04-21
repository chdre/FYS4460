import numpy as np
import shutil
import os

for runs in range(5):
    pores = np.random.randint(1, 21)

    porestring = ['p' + str(i) for i in range(1, pores + 1)]
    porestring = np.asarray(porestring, dtype=str)

    lmp_runstring = '-in in.task_i.lmp -var n_spheres %s -var config %s' % (
        pores, porestring)
    folder = os.path.join('simulations', 'pores_' + str(pores))

    try:
        print('1')
        os.makedirs(folder)
        shutil.copy('in.task_i.lmp', folder)

        print('2')
        with open('job_template.sh', 'r') as infile:
            job_string = infile.read() % lmp_runstring
            print('3')
        with open(os.path.join(folder, 'job.sh'), 'w') as outfile:
            outfile.write(job_string)
            print('4')

    except:
        pass
