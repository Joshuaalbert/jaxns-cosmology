#!/bin/bash

# experiments = [
#        ('dynesty', dynesty_models_and_parameters()),
#        ('pypolychord', pypolychord_models_and_parameters()),
#        ('nautilus', nautilus_models_and_parameters()),
#        ('pymultinest', pymultinest_models_and_parameters()),
#        ('jaxns', jaxns_models_and_parameters()),
#        ('nessai', nessai_models_and_parameters()),
#        ('ultranest', ultranest_models_and_parameters())
#    ]

for sampler_name in dynesty pypolychord nautilus pymultinest jaxns nessai ultranest; do
  python main.py $sampler_name &
done

wait
# by running 'wait' we ensure that the script will not exit until all the
# background processes have finished.