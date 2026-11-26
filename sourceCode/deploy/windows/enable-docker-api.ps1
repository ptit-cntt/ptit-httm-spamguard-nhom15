# Chạy bằng PowerShell (Run as Administrator) trên PC 192.168.1.5.
# Bước trước: Docker Desktop > Settings > General > bật
#   "Expose daemon on tcp://localhost:2375 without TLS" > Apply & restart.
#
# CẢNH BÁO: Docker API 2375 không có xác thực. Ai gọi được cổng này có toàn quyền trên máy.
# Script chỉ mở cho đúng IP server 192.168.1.222; muốn đóng lại thì chạy disable-docker-api.ps1.

$LanIp = "192.168.1.5"
$DeployServerIp = "192.168.1.222"

# Docker Desktop chỉ lắng nghe 127.0.0.1:2375 -> chuyển tiếp từ IP LAN vào
netsh interface portproxy delete v4tov4 listenaddress=$LanIp listenport=2375 2>$null
netsh interface portproxy add v4tov4 listenaddress=$LanIp listenport=2375 connectaddress=127.0.0.1 connectport=2375

Get-NetFirewallRule -DisplayName "SpamGuard*" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName "SpamGuard - Docker API (deploy server only)" -Direction Inbound `
    -Protocol TCP -LocalPort 2375 -RemoteAddress $DeployServerIp -Action Allow | Out-Null
# API cho Nginx trên server; Grafana/MLflow/Prometheus chỉ cho mạng LAN
New-NetFirewallRule -DisplayName "SpamGuard - API from Nginx server" -Direction Inbound `
    -Protocol TCP -LocalPort 8010 -RemoteAddress $DeployServerIp -Action Allow | Out-Null
New-NetFirewallRule -DisplayName "SpamGuard - Monitoring (LAN)" -Direction Inbound `
    -Protocol TCP -LocalPort 3010,5010,9095 -RemoteAddress LocalSubnet -Action Allow | Out-Null

netsh interface portproxy show v4tov4
Write-Host "Da mo Docker API cho $DeployServerIp. Kiem tra tu server: docker -H tcp://${LanIp}:2375 version"
