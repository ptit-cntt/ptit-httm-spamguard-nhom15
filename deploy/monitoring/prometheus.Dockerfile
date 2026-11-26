# Cấu hình được đóng gói vào image vì daemon Docker ở xa không bind mount được file cục bộ
FROM prom/prometheus:v2.55.1
COPY monitoring/prometheus.yml /etc/prometheus/prometheus.yml
COPY monitoring/alert_rules.yml /etc/prometheus/alert_rules.yml
