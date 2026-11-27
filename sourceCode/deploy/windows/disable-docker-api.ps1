# Đóng Docker API sau khi deploy xong (Run as Administrator). Ứng dụng vẫn chạy bình thường.
netsh interface portproxy delete v4tov4 listenaddress=192.168.1.234 listenport=2375
Get-NetFirewallRule -DisplayName "SpamGuard - Docker API*" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
Write-Host "Da dong Docker API. Nho tat 'Expose daemon on tcp://localhost:2375' trong Docker Desktop."
