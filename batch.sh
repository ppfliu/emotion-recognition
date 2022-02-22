# Copyright (C) 2020 Pengfei Liu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

for fold in {1..5}
do
  make prepare-dataset fold=$fold
  for feat in spec_mfcc
  do
    make extract-features fold=$fold feat=$feat
  done
done

feat=spec_mfcc
for fold in {1..5}
do
  mkdir -p logs-$fold
  for run in {1..5}
  do
    for source in audio text
    do
      for batch in 32
      do
        make run-experiment fold=$fold source=$source feat=$feat fusion=none batch=$batch > \
           logs-$fold/$source-fold-$fold-feat-$feat-run-$run-batch-$batch-none.log
      done
    done

    for source in audio_text
    do
      for batch in 32
      do
        for fusion in concat concat-attention concat-all ggf tfl gmu
        do
          make run-experiment fold=$fold source=$source feat=$feat fusion=$fusion batch=$batch > \
            logs-$fold/$source-fold-$fold-feat-$feat-run-$run-batch-$batch-$fusion.log
        done
      done
    done
  done
done
