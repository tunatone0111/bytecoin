#!/bin/bash
cd /home/aistartup/bytecoin
python -m microservice.bert.multicrawlandlabel_save_in2db --way timed -m 10 -b 64 -M /home/aistartup/bytecoin/microservice/bert/models/movie.tar