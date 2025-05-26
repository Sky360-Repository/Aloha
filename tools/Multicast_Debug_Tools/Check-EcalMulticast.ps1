# \copyright    Sky360.org
#
# \brief        Script to verify multicast configuration for Windows. 
#
# ********************************************************
#
# PowerShell Script to Diagnose UDP / IGMP / Multicast Issues on Windows
# Save as: Check-EcalMulticast.ps1
# Run in PowerShell as Administrator
#
# Set-ExecutionPolicy RemoteSigned -Scope Process
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\Check-EcalMulticast.ps1

function Log($msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Green
}
function Warn($msg) {
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

Log "Starting eCAL Multicast Diagnostic..."

# 1. Show active network adapters
$adapters = Get-NetIPConfiguration | Where-Object { $_.IPv4Address -ne $null }
foreach ($adapter in $adapters) {
    Log "Active Adapter: $($adapter.InterfaceAlias)"
    Log "IP Address: $($adapter.IPv4Address.IPAddress)"
    Log "Gateway: $($adapter.IPv4DefaultGateway.NextHop)"
}

# 2. Check multicast routes
Log "Checking multicast routing entries..."
$routes = Get-NetRoute | Where-Object { $_.DestinationPrefix -like "239.*" }
if ($routes) {
    foreach ($route in $routes) {
        Write-Host "  $($route.DestinationPrefix) via $($route.NextHop) on $($route.InterfaceAlias)"
    }
} else {
    Warn "No multicast routes found for 239.0.0.0/24"
    Write-Host "You may add one using:"
    Write-Host '  route -p add 239.0.0.0 mask 255.255.255.0 <Your IP Address>'
    Write-Host '  <Your IP Address> is the IP Address for you Active Adapter listed above'    
}

# 3. Check Windows Firewall status
Log "Checking Windows Firewall status..."
$firewall = (Get-NetFirewallProfile | Select-Object -Property Name, Enabled)
$firewall | Format-Table -AutoSize

# 4. Check if ICMPv4-In and UDP-In rules exist
Log "Checking firewall rules for ICMP and UDP..."
$icmpRule = Get-NetFirewallRule -DisplayName "*ICMPv4-In*" -ErrorAction SilentlyContinue | Where-Object { $_.Enabled -eq "True" }
$udpRule  = Get-NetFirewallRule -DisplayName "*UDP*" -ErrorAction SilentlyContinue | Where-Object { $_.Enabled -eq "True" }

if ($icmpRule) {
    Log "ICMPv4-In rule is enabled"
} else {
    Warn "ICMPv4-In rule is not enabled (may block ping responses)"
}

if ($udpRule) {
    Log "UDP firewall rule is enabled"
} else {
    Warn "UDP firewall rule may be missing or disabled"
    Write-Host "To add a rule:"
    Write-Host '  New-NetFirewallRule -DisplayName "eCAL UDP In" -Direction Inbound -Protocol UDP -LocalPort 14000 -Action Allow'
}

# 5. IGMP diagnostics (basic test: listing joined multicast groups)
Log "Verify that Interface is Active and Connected for IPv4 and IPv6 ..."
$groups = Get-NetIPInterface
if ($groups) {
    $groups | Format-Table -AutoSize
} else {
    Warn "No IGMP group memberships found"
    Write-Host "IGMP group join may not be working or hasn't been triggered"
}

# 6. IGMP diagnostics (basic test: listing IP joined multicast groups)
Log "Verify the IGMP group memberships: IP address is set 239.0.0.1 for your Interface"
$ip_groups = netsh interface ip show joins
if ($ip_groups) {
    $ip_groups
} else {
    Warn "No IP joined memberships"
    Write-Host "IGMP group join may not be working or hasn't been triggered"
}





Log "`nDiagnostics complete."
