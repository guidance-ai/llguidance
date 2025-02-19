#!/bin/sh

# random 100 files
BENCH_FILES="Github_medium---o79016.json Github_easy---o69997.json Github_medium---o24621.json Glaiveai2K---create_invoice_84ada69d.json Github_medium---o9965.json Github_medium---o14473.json Github_medium---o10078.json Glaiveai2K---search_news_a15c8733.json JsonSchemaStore---ansible-navigator.json Glaiveai2K---calculate_area_51f69312.json Github_hard---o41193.json Github_hard---o41345.json Snowplow---sp_183_Normalized.json Glaiveai2K---calculate_area_aeb5bb1c.json Snowplow---sp_397_Normalized.json Github_hard---o47195.json Github_easy---o43995.json Github_ultra---o21312.json Github_medium---o83835.json Github_trivial---o60137.json Github_medium---o83992.json Glaiveai2K---calculate_area_7ab05fe7.json Github_medium---o48008.json Glaiveai2K---calculate_area_c25c3534.json Github_hard---o53084.json Github_trivial---o90334.json Github_medium---o87934.json Kubernetes---kb_444_Normalized.json Snowplow---sp_245_Normalized.json Kubernetes---kb_1112_Normalized.json Github_hard---o53128.json Glaiveai2K---schedule_meeting_b3365712.json Github_medium---o3518.json Github_easy---o90248.json Github_hard---o41372.json Glaiveai2K---analyze_social_media_mentions_df9069d1.json Kubernetes---kb_84_Normalized.json Glaiveai2K---calculate_area_eb46c7e3.json Github_medium---o9906.json Glaiveai2K---calculate_area_45887aa5.json Glaiveai2K---calculate_area_volume_21b6fca7.json Github_easy---o12078.json Github_hard---o7514.json Github_ultra---o6362.json JsonSchemaStore---bukkit-plugin.json Github_easy---o27037.json Kubernetes---kb_579_Normalized.json Glaiveai2K---calculate_carbon_footprint_46a93a9d.json Github_trivial---o83137.json Github_easy---o21053.json Github_easy---o63674.json Github_easy---o17683.json Glaiveai2K---calculate_area_59aeb8e9.json Github_easy---o21114.json Github_easy---o20489.json Github_hard---o30263.json Glaiveai2K---calculate_discounted_price_ac4734a9.json Github_easy---o13671.json Github_medium---o6189.json Github_medium---o9510.json Github_easy---o37095.json Github_hard---o54898.json Kubernetes---kb_152_Normalized.json Github_trivial---o45592.json Github_easy---o71444.json Github_hard---o58446.json Github_easy---o61120.json Github_hard---o68416.json Kubernetes---kb_1139_Normalized.json Github_hard---o82738.json Github_hard---o6084.json Github_medium---o44349.json Github_medium---o9520.json Github_trivial---o45029.json Github_trivial---o41609.json Github_medium---o9359.json Github_hard---o83677.json Glaiveai2K---calculate_area_bf611aaf.json Glaiveai2K---calculate_shipping_cost_34a93e55.json Github_hard---o59215.json Github_easy---o65305.json Handwritten---oneof5_2.json Github_medium---o9940.json Github_easy---o55691.json Github_medium---o73951.json Github_hard---o12337.json Glaiveai2K---calculate_area_143516bf.json Glaiveai2K---calculate_area_771478b4.json Github_hard---o47199.json Kubernetes---kb_678_Normalized.json Snowplow---sp_94_Normalized.json Github_easy---o45388.json Github_easy---o6250.json JsonSchemaStore---fossa-yml.v3.schema.json Kubernetes---kb_1127_Normalized.json Kubernetes---kb_453_Normalized.json JsonSchemaStore---sarif-2.1.0-rtm.4.json Github_medium---o46359.json Github_hard---o73113.json Github_medium---o89504.json"

for folder in ../.. .. ../../tmp ; do
    if test -d $folder/jsonschemabench/maskbench/data; then
        MB_DATA=$folder/jsonschemabench/maskbench/data/
        break
    fi
done

if [ "$1" == "--bench" ] ; then
    shift
    DEFAULT_ARGS="-m --expected expected_maskbench.json --ballpark --num-threads 1 "
    for f in $BENCH_FILES; do
        DEFAULT_ARGS="$DEFAULT_ARGS $MB_DATA/$f"
    done
fi

if [ -z "$PERF" ]; then
    cargo build --release
    ../target/release/json_stats $DEFAULT_ARGS "$@"
else
    PERF='perf record -F 9999 -g'
    RUSTFLAGS='-C force-frame-pointers=y' cargo build --profile perf
    $PERF ../target/perf/json_stats $DEFAULT_ARGS "$@"
    echo "perf report -g graph,0.05,caller"
    echo "perf report -g graph,0.05,callee"
fi

