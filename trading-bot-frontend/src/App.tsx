import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Copy, CheckCircle2, XCircle, TrendingUp } from 'lucide-react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface Trade {
  timestamp: string
  symbol: string
  coin: string
  side: string
  amount: number
  entry_price: number
  order_id: string
  status: string
  stop_loss?: string
  take_profit?: string
  pnl: number
}

interface WebhookLog {
  timestamp: string
  payload: any
  status: string
  executed: boolean
  error: string | null
}

interface Status {
  api_connected: boolean
  exchange: string | null
  auto_trading_enabled: boolean
  emergency_stop: boolean
  buy_amount_usd: number
  coin_trading_enabled: {
    BTC: boolean
    ETH: boolean
    SOL: boolean
    XRP: boolean
  }
  last_webhook: {
    timestamp: string
    payload: any
    test?: boolean
  } | null
  last_order: {
    timestamp: string
    symbol?: string
    side?: string
    amount?: number
    price?: string
    order_id?: string
    status?: string
    error?: string
  } | null
  pnl_summary: {
    total_orders: number
    total_pnl: number
    per_coin_pnl: { [key: string]: number }
    recent_orders: any[]
    winning_trades: number
    losing_trades: number
    win_rate: number
    avg_trade_size: number
    total_exposure: number
  }
  open_trades: Trade[]
  closed_trades: Trade[]
  webhook_logs: WebhookLog[]
}

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [authToken, setAuthToken] = useState<string | null>(null)
  const [showRegister, setShowRegister] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [currentUsername, setCurrentUsername] = useState<string | null>(null)
  const [apiKey, setApiKey] = useState('')
  const [apiSecret, setApiSecret] = useState('')
  const [exchange, setExchange] = useState('binance')
  const [webhookUrl, setWebhookUrl] = useState('')
  const [autoTrading, setAutoTrading] = useState(false)
  const [buyAmount, setBuyAmount] = useState(5)
  const [status, setStatus] = useState<Status | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)
  const [copied, setCopied] = useState(false)

  const formatTimestamp = (timestamp: string, options?: Intl.DateTimeFormatOptions) => {
    const date = new Date(timestamp)
    return date.toLocaleString('en-US', {
      timeZone: 'America/New_York',
      dateStyle: 'medium',
      timeStyle: 'short',
      ...options
    })
  }

  useEffect(() => {
    const token = localStorage.getItem('authToken')
    const savedUsername = localStorage.getItem('username')
    if (token && savedUsername) {
      setAuthToken(token)
      setCurrentUsername(savedUsername)
      setIsAuthenticated(true)
    }
  }, [])

  useEffect(() => {
    if (isAuthenticated && authToken) {
      fetchStatus()
      fetchWebhookUrl()
      const interval = setInterval(fetchStatus, 3000)
      return () => clearInterval(interval)
    }
  }, [isAuthenticated, authToken])

  const handleRegister = async () => {
    if (!username || !password) {
      setMessage({ type: 'error', text: 'Please enter username and password' })
      return
    }

    if (username.length < 3) {
      setMessage({ type: 'error', text: 'Username must be at least 3 characters' })
      return
    }

    if (password.length < 8) {
      setMessage({ type: 'error', text: 'Password must be at least 8 characters' })
      return
    }

    if (password !== confirmPassword) {
      setMessage({ type: 'error', text: 'Passwords do not match' })
      return
    }

    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${API_URL}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      })

      const data = await response.json()

      if (response.ok) {
        setMessage({ type: 'success', text: 'Registration successful! Please login.' })
        setShowRegister(false)
        setPassword('')
        setConfirmPassword('')
      } else {
        setMessage({ type: 'error', text: data.detail || 'Registration failed' })
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to connect to backend' })
    } finally {
      setLoading(false)
    }
  }

  const handleLogin = async () => {
    if (!username || !password) {
      setMessage({ type: 'error', text: 'Please enter username and password' })
      return
    }

    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${API_URL}/user/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      })

      const data = await response.json()

      if (response.ok) {
        setAuthToken(data.token)
        setCurrentUsername(data.username)
        setIsAuthenticated(true)
        localStorage.setItem('authToken', data.token)
        localStorage.setItem('username', data.username)
        setMessage({ type: 'success', text: 'Login successful!' })
      } else {
        setMessage({ type: 'error', text: data.detail || 'Invalid username or password' })
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to connect to backend' })
    } finally {
      setLoading(false)
    }
  }

  const fetchStatus = async () => {
    if (!authToken) return
    
    try {
      const response = await fetch(`${API_URL}/status`, {
        headers: { 'Authorization': `Bearer ${authToken}` }
      })
      
      if (response.status === 401) {
        setIsAuthenticated(false)
        localStorage.removeItem('authToken')
        return
      }
      
      const data = await response.json()
      setStatus(data)
      setAutoTrading(data.auto_trading_enabled)
      setBuyAmount(data.buy_amount_usd || 5)
    } catch (error) {
      console.error('Failed to fetch status:', error)
    }
  }

  const fetchWebhookUrl = async () => {
    if (!authToken) return
    
    try {
      const response = await fetch(`${API_URL}/webhook-url`, {
        headers: { 'Authorization': `Bearer ${authToken}` }
      })
      
      if (response.status === 401) {
        setIsAuthenticated(false)
        localStorage.removeItem('authToken')
        return
      }
      
      const data = await response.json()
      setWebhookUrl(data.webhook_url)
    } catch (error) {
      console.error('Failed to fetch webhook URL:', error)
    }
  }

  const handleSaveSettings = async () => {
    if (!apiKey || !apiSecret) {
      setMessage({ type: 'error', text: 'Please enter both API Key and API Secret' })
      return
    }

    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${API_URL}/api/settings`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({
          apiKey: apiKey,
          apiSecret: apiSecret,
          exchange: exchange
        })
      })

      const data = await response.json()

      if (response.ok) {
        setMessage({ type: 'success', text: data.message || 'API credentials saved and verified successfully!' })
        fetchStatus()
      } else {
        setMessage({ type: 'error', text: data.detail || 'Failed to save API credentials' })
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to connect to backend' })
    } finally {
      setLoading(false)
    }
  }

  const handleToggleTrading = async (enabled: boolean) => {
    try {
      const response = await fetch(`${API_URL}/toggle-trading`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({ enabled, buyAmount })
      })

      if (response.ok) {
        setAutoTrading(enabled)
        setMessage({ 
          type: 'success', 
          text: enabled ? 'Auto-trading enabled!' : 'Auto-trading disabled!' 
        })
        fetchStatus()
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to toggle auto-trading' })
    }
  }

  const handleUpdateBuyAmount = async (amount: number) => {
    try {
      const response = await fetch(`${API_URL}/update-buy-amount`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({ buy_amount_usd: amount })
      })

      if (response.ok) {
        const data = await response.json()
        setBuyAmount(data.buy_amount_usd)
        setMessage({ 
          type: 'success', 
          text: `Buy amount updated to $${data.buy_amount_usd}` 
        })
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to update buy amount' })
    }
  }

  const handleTestWebhook = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/test-webhook`, {
        method: 'POST',
        headers: { 
          'Authorization': `Bearer ${authToken}`
        }
      })

      if (response.ok) {
        setMessage({ type: 'success', text: 'Test webhook sent successfully!' })
        fetchStatus()
      } else {
        setMessage({ type: 'error', text: 'Failed to send test webhook' })
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to connect to backend' })
    } finally {
      setLoading(false)
    }
  }

  const handleCopyWebhook = () => {
    navigator.clipboard.writeText(webhookUrl)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleToggleCoin = async (coin: string, enabled: boolean) => {
    try {
      const response = await fetch(`${API_URL}/toggle-coin`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({ coin, enabled })
      })

      if (response.ok) {
        setMessage({ 
          type: 'success', 
          text: `${coin} trading ${enabled ? 'enabled' : 'disabled'}!` 
        })
        fetchStatus()
      }
    } catch (error) {
      setMessage({ type: 'error', text: `Failed to toggle ${coin} trading` })
    }
  }

  const handleEmergencyStop = async () => {
    try {
      const response = await fetch(`${API_URL}/emergency-stop`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({ stop: true })
      })

      if (response.ok) {
        setMessage({ 
          type: 'success', 
          text: 'EMERGENCY STOP ACTIVATED - All trading disabled!' 
        })
        fetchStatus()
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to activate emergency stop' })
    }
  }

  const handleResumeTrading = async () => {
    try {
      const response = await fetch(`${API_URL}/emergency-stop`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({ stop: false })
      })

      if (response.ok) {
        setMessage({ 
          type: 'success', 
          text: 'Emergency stop deactivated - Trading can be resumed' 
        })
        fetchStatus()
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to deactivate emergency stop' })
    }
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-6">
        <Card className="w-full max-w-md bg-slate-800 border-slate-700">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl font-bold text-white mb-2">Trading Bot</CardTitle>
            <CardDescription className="text-slate-400">
              {showRegister ? 'Create a new account' : 'Login to access dashboard'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {message && (
              <Alert className={message.type === 'success' ? 'bg-green-900/20 border-green-500' : 'bg-red-900/20 border-red-500'}>
                <AlertDescription className={message.type === 'success' ? 'text-green-400' : 'text-red-400'}>
                  {message.text}
                </AlertDescription>
              </Alert>
            )}
            
            <div className="space-y-2">
              <Label htmlFor="username" className="text-slate-300">Username</Label>
              <Input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                className="bg-slate-700 border-slate-600 text-white placeholder:text-slate-500"
                autoFocus
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-slate-300">Password</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && (showRegister ? handleRegister() : handleLogin())}
                placeholder="Enter password"
                className="bg-slate-700 border-slate-600 text-white placeholder:text-slate-500"
              />
            </div>

            {showRegister && (
              <div className="space-y-2">
                <Label htmlFor="confirmPassword" className="text-slate-300">Confirm Password</Label>
                <Input
                  id="confirmPassword"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleRegister()}
                  placeholder="Confirm password"
                  className="bg-slate-700 border-slate-600 text-white placeholder:text-slate-500"
                />
              </div>
            )}

            <Button 
              onClick={showRegister ? handleRegister : handleLogin} 
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700"
            >
              {loading ? (showRegister ? 'Registering...' : 'Logging in...') : (showRegister ? 'Register' : 'Login')}
            </Button>

            <div className="text-center">
              <button
                onClick={() => {
                  setShowRegister(!showRegister)
                  setMessage(null)
                  setPassword('')
                  setConfirmPassword('')
                }}
                className="text-blue-400 hover:text-blue-300 text-sm"
              >
                {showRegister ? 'Already have an account? Login' : "Don't have an account? Register"}
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  const handleLogout = () => {
    setIsAuthenticated(false)
    setAuthToken(null)
    setCurrentUsername(null)
    localStorage.removeItem('authToken')
    localStorage.removeItem('username')
    setMessage({ type: 'success', text: 'Logged out successfully' })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex justify-between items-center mb-8">
          <div className="text-center flex-1">
            <h1 className="text-4xl font-bold text-white mb-2">Advanced Trading Bot Dashboard</h1>
            <p className="text-slate-400">Automated TradingView Alert Execution with Full Control</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-sm text-slate-400">Logged in as</p>
              <p className="text-white font-medium">{currentUsername}</p>
            </div>
            <Button 
              onClick={handleLogout}
              className="bg-slate-700 hover:bg-slate-600"
            >
              Logout
            </Button>
          </div>
        </div>

        {message && (
          <Alert className={message.type === 'success' ? 'bg-green-900/20 border-green-500' : 'bg-red-900/20 border-red-500'}>
            <AlertDescription className={message.type === 'success' ? 'text-green-400' : 'text-red-400'}>
              {message.text}
            </AlertDescription>
          </Alert>
        )}

        {status?.emergency_stop && (
          <Alert className="bg-red-900/30 border-red-500">
            <AlertDescription className="text-red-400 font-bold text-center">
              EMERGENCY STOP ACTIVE - All trading is disabled!
            </AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Emergency Controls</CardTitle>
              <CardDescription className="text-slate-400">Quick stop and resume trading</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {status?.emergency_stop ? (
                <Button 
                  onClick={handleResumeTrading}
                  className="w-full bg-green-600 hover:bg-green-700"
                >
                  Resume Trading
                </Button>
              ) : (
                <Button 
                  onClick={handleEmergencyStop}
                  className="w-full bg-red-600 hover:bg-red-700"
                >
                  EMERGENCY STOP
                </Button>
              )}
              
              <div className="space-y-2">
                <Label className="text-slate-300">Auto-Trading</Label>
                <div className="flex items-center justify-between p-4 bg-slate-700 rounded-lg">
                  <span className="text-white font-medium">
                    {status?.auto_trading_enabled ? 'Enabled' : 'Disabled'}
                  </span>
                  <Switch
                    checked={status?.auto_trading_enabled || false}
                    onCheckedChange={handleToggleTrading}
                    disabled={status?.emergency_stop}
                    className="data-[state=checked]:bg-green-600"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-slate-300">Buy Amount per Trade</Label>
                <div className="p-4 bg-slate-700 rounded-lg space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-white font-medium text-lg">
                      ${buyAmount.toLocaleString()}
                    </span>
                    <span className="text-slate-400 text-sm">
                      ${buyAmount === 5 ? '5' : buyAmount.toLocaleString()} USD
                    </span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="100000"
                    step="5"
                    value={buyAmount}
                    onChange={(e) => {
                      const newAmount = Number(e.target.value)
                      setBuyAmount(newAmount)
                    }}
                    onMouseUp={(e) => {
                      const newAmount = Number((e.target as HTMLInputElement).value)
                      handleUpdateBuyAmount(newAmount)
                    }}
                    onTouchEnd={(e) => {
                      const newAmount = Number((e.target as HTMLInputElement).value)
                      handleUpdateBuyAmount(newAmount)
                    }}
                    disabled={status?.emergency_stop}
                    className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer slider"
                    style={{
                      background: `linear-gradient(to right, #10b981 0%, #10b981 ${((buyAmount - 5) / (100000 - 5)) * 100}%, #475569 ${((buyAmount - 5) / (100000 - 5)) * 100}%, #475569 100%)`
                    }}
                  />
                  <div className="flex justify-between text-xs text-slate-400">
                    <span>$5</span>
                    <span>$100,000</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Coin Trading Controls</CardTitle>
              <CardDescription className="text-slate-400">Enable/disable individual coins</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {status?.coin_trading_enabled && Object.entries(status.coin_trading_enabled).map(([coin, enabled]) => (
                <div key={coin} className="flex items-center justify-between p-3 bg-slate-700 rounded-lg">
                  <span className="text-white font-medium">{coin}</span>
                  <Switch
                    checked={enabled}
                    onCheckedChange={(checked) => handleToggleCoin(coin, checked)}
                    disabled={status?.emergency_stop}
                    className="data-[state=checked]:bg-green-600"
                  />
                </div>
              ))}
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Risk Management</CardTitle>
              <CardDescription className="text-slate-400">Current exposure and limits</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="bg-slate-700 p-3 rounded-lg">
                <div className="text-slate-400 text-sm">Total Exposure</div>
                <div className="text-xl font-bold text-white">
                  ${status?.pnl_summary.total_exposure?.toFixed(2) || '0.00'}
                </div>
              </div>
              <div className="bg-slate-700 p-3 rounded-lg">
                <div className="text-slate-400 text-sm">Avg Trade Size</div>
                <div className="text-xl font-bold text-white">
                  {status?.pnl_summary.avg_trade_size?.toFixed(4) || '0.0000'}
                </div>
              </div>
              <div className="bg-slate-700 p-3 rounded-lg">
                <div className="text-slate-400 text-sm">Position Sizing</div>
                <div className="text-sm text-white">2% of balance</div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="bg-slate-800 border border-slate-700 p-4 rounded-lg">
            <div className="text-slate-400 text-sm mb-1">API Connection</div>
            <div className="flex items-center gap-2">
              {status?.api_connected ? (
                <>
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                  <span className="text-white font-medium">Connected</span>
                </>
              ) : (
                <>
                  <XCircle className="h-5 w-5 text-red-500" />
                  <span className="text-white font-medium">Not Connected</span>
                </>
              )}
            </div>
            {status?.exchange && (
              <Badge className="mt-2 bg-blue-600">{status.exchange}</Badge>
            )}
          </div>

          <div className="bg-slate-800 border border-slate-700 p-4 rounded-lg">
            <div className="text-slate-400 text-sm mb-1">Total Orders</div>
            <div className="text-2xl font-bold text-white">
              {status?.pnl_summary.total_orders || 0}
            </div>
          </div>

          <div className="bg-slate-800 border border-slate-700 p-4 rounded-lg">
            <div className="text-slate-400 text-sm mb-1">Win Rate</div>
            <div className="text-2xl font-bold text-white">
              {status?.pnl_summary.win_rate?.toFixed(1) || '0.0'}%
            </div>
            <div className="text-xs text-slate-400 mt-1">
              {status?.pnl_summary.winning_trades || 0}W / {status?.pnl_summary.losing_trades || 0}L
            </div>
          </div>

          <div className="bg-slate-800 border border-slate-700 p-4 rounded-lg">
            <div className="text-slate-400 text-sm mb-1 flex items-center gap-1">
              <TrendingUp className="h-4 w-4" />
              Total PnL
            </div>
            <div className="text-2xl font-bold text-white">
              ${status?.pnl_summary.total_pnl?.toFixed(2) || '0.00'}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">API Configuration</CardTitle>
              <CardDescription className="text-slate-400">Connect your exchange account</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="exchange" className="text-slate-300">Exchange</Label>
                <Select value={exchange} onValueChange={setExchange}>
                  <SelectTrigger id="exchange" className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-700 border-slate-600">
                    <SelectItem value="binance">Binance</SelectItem>
                    <SelectItem value="kucoin">KuCoin</SelectItem>
                    <SelectItem value="kraken">Kraken</SelectItem>
                    <SelectItem value="bybit">Bybit</SelectItem>
                    <SelectItem value="coinbase">Coinbase Advanced</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="apiKey" className="text-slate-300">API Key</Label>
                <Input
                  id="apiKey"
                  type="text"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Enter your API key"
                  className="bg-slate-700 border-slate-600 text-white placeholder:text-slate-500"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="apiSecret" className="text-slate-300">API Secret</Label>
                <Input
                  id="apiSecret"
                  type="password"
                  value={apiSecret}
                  onChange={(e) => setApiSecret(e.target.value)}
                  placeholder="Enter your API secret"
                  className="bg-slate-700 border-slate-600 text-white placeholder:text-slate-500"
                />
              </div>

              <Button 
                onClick={handleSaveSettings} 
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-700"
              >
                {loading ? 'Saving...' : 'Save Settings'}
              </Button>
            </CardContent>
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Webhook Configuration</CardTitle>
              <CardDescription className="text-slate-400">Copy this URL to TradingView</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label className="text-slate-300">Webhook URL</Label>
                <div className="flex gap-2">
                  <Input
                    value={webhookUrl}
                    readOnly
                    className="bg-slate-700 border-slate-600 text-white font-mono text-sm"
                  />
                  <Button
                    onClick={handleCopyWebhook}
                    variant="outline"
                    className="border-slate-600 hover:bg-slate-700"
                  >
                    {copied ? <CheckCircle2 className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-slate-300">Auto-Trading</Label>
                <div className="flex items-center justify-between p-4 bg-slate-700 rounded-lg">
                  <span className="text-white font-medium">
                    {autoTrading ? 'Enabled' : 'Disabled'}
                  </span>
                  <Switch
                    checked={autoTrading}
                    onCheckedChange={handleToggleTrading}
                    className="data-[state=checked]:bg-green-600"
                  />
                </div>
              </div>

              <Button 
                onClick={handleTestWebhook}
                disabled={loading}
                variant="outline"
                className="w-full border-slate-600 hover:bg-slate-700 text-white"
              >
                Test Webhook
              </Button>
            </CardContent>
          </Card>
        </div>

        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Open Trades</CardTitle>
            <CardDescription className="text-slate-400">Active positions with live P&L</CardDescription>
          </CardHeader>
          <CardContent>
            {status?.open_trades && status.open_trades.length > 0 ? (
              <div className="space-y-3">
                {status.open_trades.map((trade, idx) => (
                  <div key={idx} className="bg-slate-700 p-4 rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <div className="text-white font-bold">{trade.symbol}</div>
                        <div className="text-xs text-slate-400">{formatTimestamp(trade.timestamp)}</div>
                      </div>
                      <Badge className="bg-green-600">{trade.side}</Badge>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-sm">
                      <div>
                        <div className="text-slate-400">Amount</div>
                        <div className="text-white">{trade.amount?.toFixed(4)}</div>
                      </div>
                      <div>
                        <div className="text-slate-400">Entry Price</div>
                        <div className="text-white">${trade.entry_price?.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-slate-400">P&L</div>
                        <div className={trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                          ${trade.pnl?.toFixed(2)}
                        </div>
                      </div>
                    </div>
                    {(trade.stop_loss || trade.take_profit) && (
                      <div className="mt-2 text-xs text-slate-400">
                        {trade.stop_loss && <span>SL: ${trade.stop_loss} </span>}
                        {trade.take_profit && <span>TP: ${trade.take_profit}</span>}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-slate-400 text-center py-8">No open trades</div>
            )}
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Closed Trades</CardTitle>
            <CardDescription className="text-slate-400">Recent trade history</CardDescription>
          </CardHeader>
          <CardContent>
            {status?.closed_trades && status.closed_trades.length > 0 ? (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {status.closed_trades.map((trade, idx) => (
                  <div key={idx} className="bg-slate-700 p-3 rounded-lg">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <Badge className={trade.side === 'buy' ? 'bg-green-600' : 'bg-red-600'}>
                          {trade.side}
                        </Badge>
                        <span className="text-white font-medium">{trade.symbol}</span>
                        <span className="text-slate-400 text-sm">{trade.amount?.toFixed(4)}</span>
                      </div>
                      <div className="text-right">
                        <div className={trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                          ${trade.pnl?.toFixed(2)}
                        </div>
                        <div className="text-xs text-slate-400">
                          {formatTimestamp(trade.timestamp, { dateStyle: undefined, timeStyle: 'short' })}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-slate-400 text-center py-8">No closed trades yet</div>
            )}
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Webhook Logs</CardTitle>
            <CardDescription className="text-slate-400">Recent webhook activity and status</CardDescription>
          </CardHeader>
          <CardContent>
            {status?.webhook_logs && status.webhook_logs.length > 0 ? (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {status.webhook_logs.map((log, idx) => (
                  <div key={idx} className="bg-slate-700 p-3 rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <div className="text-xs text-slate-400">
                        {formatTimestamp(log.timestamp)}
                      </div>
                      <Badge className={
                        log.status === 'executed' ? 'bg-green-600' :
                        log.status === 'rejected' ? 'bg-red-600' :
                        log.status === 'skipped' ? 'bg-yellow-600' :
                        'bg-slate-600'
                      }>
                        {log.status}
                      </Badge>
                    </div>
                    <div className="text-sm text-slate-300">
                      {log.payload?.action} {log.payload?.symbol}
                    </div>
                    {log.error && (
                      <div className="text-xs text-red-400 mt-1">{log.error}</div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-slate-400 text-center py-8">No webhook logs yet</div>
            )}
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">TradingView Alert Setup</CardTitle>
            <CardDescription className="text-slate-400">Use this JSON format in your TradingView alerts</CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="bg-slate-900 p-4 rounded-lg overflow-auto text-sm text-slate-300">
{`{
  "secret": "Samkremer12",
  "action": "buy",
  "symbol": "BTCUSDT",
  "price": "{{close}}",
  "quantity": 0.001
}`}
            </pre>
            <div className="mt-4 text-sm text-slate-400 space-y-1">
              <p>• <span className="text-white">secret</span>: "Samkremer12" (required for security)</p>
              <p>• <span className="text-white">action</span>: "buy", "sell", "long", "short", or "close"</p>
              <p>• <span className="text-white">symbol</span>: "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"</p>
              <p>• <span className="text-white">price</span>: Use TradingView variables like {`{{close}}`} or specific price</p>
              <p>• <span className="text-white">quantity</span>: Amount to trade (optional, auto-calculated at 2% of balance)</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
