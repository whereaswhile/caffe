#!/bin/bash

jobn=10
testset=val
testtask=loc+cf #loc
outfolder="./fea/pred-${testset}/"

if (( 1==1 ))
then
	rm $outfolder*.job*

	#for i in $(seq 0 1 ${jobn} )
	for (( i=0; i<${jobn}; i++ ))
	do
		echo "submitting job ${i}..."
		python generate_pred_bbx_${testset}-${testtask}.py ${jobn} ${i} &
		pids[$i]=$!
	done

	for (( i=0; i<${jobn}; i++ ))
	do
		echo "waiting for pid ${pids[$i]}"
		wait ${pids[$i]}
	done
fi

fn1=`ls ${outfolder}pred1*job0`
fn1=${fn1%.job0}
fn2=`ls ${outfolder}pred2*job0`
fn2=${fn2%.job0}

echo merging $fn1
echo merging $fn2
rm $fn1
rm $fn2
for (( i=0; i<${jobn}; i++ ))
do
	cat ${fn1}.job${i} >> ${fn1}
	cat ${fn2}.job${i} >> ${fn2}
done

echo "matlab evaluating..."
matlab -nodesktop -nodisplay -r "addpath './routine/'; run_eval('${fn1}'); clear; run_eval('${fn2}'); exit;"

