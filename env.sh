if [[ -d /lustre ]]; then
    export DS_DIR='/lustre/user/jackmin/datasets'
else
    export DS_DIR='/data/datasets'
fi
export RAW_DIR="$DS_DIR/ai4code"
export PROC_DIR="$DS_DIR/proc-ai4code-20pct"
export PCT_DATA=0.2

clearenv () {
    unset RAW_DIR
    unset PROC_DIR
    unset PCT_DATA
}