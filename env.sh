export DS_DIR='/data/datasets'

export RAW_DIR="$DS_DIR/ai4code"
export PROC_DIR="$DS_DIR/proc-ai4code"
export PCT_DATA=1

clearenv () {
    unset RAW_DIR
    unset PROC_DIR
    unset PCT_DATA
}