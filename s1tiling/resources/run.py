#!/bin/python

#
# Script de lancement de job pour S1Tiling
#
# Auteur: Thierry KOLECK (CNES)
#

import datetime
import os

############################
# Parametres à configurer
############################

# Date de début des données à traiter
begin_date = "2020-01-01"

# Date de fin des données à traiter
end_date = "2020-08-01"

# Nombre de jours à traiter pour chaque job
duration_per_job = 20

# Nombre de job à executer en parallèle
max_nb_jobs = 20

# Préfixe du nom du job
job_ID_prefix = "VN"

# Job à exécuter
jobname_to_run = "jobVietnam"


####################################
# Debut du script (ne pas modifier)
####################################

b_date = datetime.date.fromisoformat(begin_date)
e_date = datetime.date.fromisoformat(end_date)
d_date = datetime.timedelta(days=duration_per_job)

d = b_date

job_list = []

while d < e_date:

    if d + d_date > e_date:
        l_date = e_date
    else:
        l_date = d + d_date

    jobID = job_ID_prefix + "-" + d.strftime('%y-%m-%d')
    cmd = ("export f_date=" + d.strftime('%Y-%m-%d') + ";export l_date=" + (l_date).strftime('%Y-%m-%d'))

    if len(job_list) >= max_nb_jobs:
        cmd = cmd + ";qsub -W depend=after:" + job_list[-max_nb_jobs] + " -N " + jobID + " -v f_date,l_date " + jobname_to_run
    else:
        cmd = cmd + ";qsub -N " + jobID + " -v f_date,l_date " + jobname_to_run

    c = os.popen(cmd)
    ret = c.read()[:-1]
    job_list.append(ret)
    print(cmd)
    print(ret)

    d = d + d_date + datetime.timedelta(days=1)
