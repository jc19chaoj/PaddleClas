set -e

OUTPUT_DIR=$1
PID=$(head -n 1 $OUTPUT_DIR/alt.pid)
CUPTI_OUT=$(find $OUTPUT_DIR -name "CUptiTracer_PID*")


echo "Step 1:"
echo "    python3.7 tools/perf_analyze2.py -p $OUTPUT_DIR/$PID"
python3.7 tools/perf_analyze2.py -p $OUTPUT_DIR/$PID

cd postprocess

echo "Step 2:"
echo "    python3.7 analyze_CUptiTracer.py -p $CUPTI_OUT"
python3.7 analyze_CUptiTracer.py -p $CUPTI_OUT $2

echo "Step 3:"
echo "    python3.7 merge_json2csv.py"
python3.7 merge_json2csv.py



