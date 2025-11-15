import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Copy, CheckCircle2, XCircle, Activity, TrendingUp } from 'lucide-react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface Status {
  api_connected: boolean
  exchange: string | null
  auto_trading_enabled: boolean
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
    recent_orders: any[]
  }
}

function App() {
  const [apiKey, setApiKey] = useState('')
  const [apiSecret, setApiSecret] = useState('')
  const [exchange, setExchange] = useState('binance')
  const [webhookUrl, setWebhookUrl] = useState('')
  const [autoTrading, setAutoTrading] = useState(false)
  const [status, setStatus] = useState<Status | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    fetchStatus()
    fetchWebhookUrl()
    const interval = setInterval(fetchStatus, 3000)
    return () => clearInterval(interval)
  }, [])

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/status`)
      const data = await response.json()
      setStatus(data)
      setAutoTrading(data.auto_trading_enabled)
    } catch (error) {
      console.error('Failed to fetch status:', error)
    }
  }

  const fetchWebhookUrl = async () => {
    try {
      const response = await fetch(`${API_URL}/webhook-url`)
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
      const response = await fetch(`${API_URL}/set-api-key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: apiKey,
          api_secret: apiSecret,
          exchange: exchange
        })
      })

      const data = await response.json()

      if (response.ok) {
        setMessage({ type: 'success', text: 'API credentials saved and verified successfully!' })
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
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

  const handleTestWebhook = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/test-webhook`, {
        method: 'POST'
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Trading Bot Dashboard</h1>
          <p className="text-slate-400">Automated TradingView Alert Execution</p>
        </div>

        {message && (
          <Alert className={message.type === 'success' ? 'bg-green-900/20 border-green-500' : 'bg-red-900/20 border-red-500'}>
            <AlertDescription className={message.type === 'success' ? 'text-green-400' : 'text-red-400'}>
              {message.text}
            </AlertDescription>
          </Alert>
        )}

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
            <CardTitle className="text-white flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Status Panel
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-slate-700 p-4 rounded-lg">
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

              <div className="bg-slate-700 p-4 rounded-lg">
                <div className="text-slate-400 text-sm mb-1">Auto-Trading</div>
                <div className="flex items-center gap-2">
                  {status?.auto_trading_enabled ? (
                    <>
                      <CheckCircle2 className="h-5 w-5 text-green-500" />
                      <span className="text-white font-medium">Active</span>
                    </>
                  ) : (
                    <>
                      <XCircle className="h-5 w-5 text-yellow-500" />
                      <span className="text-white font-medium">Inactive</span>
                    </>
                  )}
                </div>
              </div>

              <div className="bg-slate-700 p-4 rounded-lg">
                <div className="text-slate-400 text-sm mb-1">Total Orders</div>
                <div className="text-2xl font-bold text-white">
                  {status?.pnl_summary.total_orders || 0}
                </div>
              </div>

              <div className="bg-slate-700 p-4 rounded-lg">
                <div className="text-slate-400 text-sm mb-1 flex items-center gap-1">
                  <TrendingUp className="h-4 w-4" />
                  Total PnL
                </div>
                <div className="text-2xl font-bold text-white">
                  ${status?.pnl_summary.total_pnl.toFixed(2) || '0.00'}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="bg-slate-700 p-4 rounded-lg">
                <h3 className="text-white font-medium mb-2">Last Webhook Received</h3>
                {status?.last_webhook ? (
                  <div className="space-y-2">
                    <div className="text-sm text-slate-400">
                      {new Date(status.last_webhook.timestamp).toLocaleString()}
                    </div>
                    <pre className="text-xs bg-slate-800 p-2 rounded overflow-auto text-slate-300">
                      {JSON.stringify(status.last_webhook.payload, null, 2)}
                    </pre>
                    {status.last_webhook.test && (
                      <Badge className="bg-yellow-600">Test Webhook</Badge>
                    )}
                  </div>
                ) : (
                  <div className="text-slate-400 text-sm">No webhook received yet</div>
                )}
              </div>

              <div className="bg-slate-700 p-4 rounded-lg">
                <h3 className="text-white font-medium mb-2">Last Order Executed</h3>
                {status?.last_order ? (
                  <div className="space-y-2">
                    <div className="text-sm text-slate-400">
                      {new Date(status.last_order.timestamp).toLocaleString()}
                    </div>
                    {status.last_order.error ? (
                      <div className="text-sm text-red-400">
                        Error: {status.last_order.error}
                      </div>
                    ) : (
                      <div className="space-y-1 text-sm text-slate-300">
                        <div>Symbol: <span className="text-white font-medium">{status.last_order.symbol}</span></div>
                        <div>Side: <Badge className={status.last_order.side === 'buy' ? 'bg-green-600' : 'bg-red-600'}>{status.last_order.side}</Badge></div>
                        <div>Amount: <span className="text-white font-medium">{status.last_order.amount}</span></div>
                        {status.last_order.price && (
                          <div>Price: <span className="text-white font-medium">${status.last_order.price}</span></div>
                        )}
                        {status.last_order.order_id && (
                          <div className="text-xs text-slate-400">Order ID: {status.last_order.order_id}</div>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-slate-400 text-sm">No orders executed yet</div>
                )}
              </div>
            </div>
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
  "action": "buy",
  "symbol": "BTCUSDT",
  "price": "{{close}}",
  "quantity": 0.001
}`}
            </pre>
            <div className="mt-4 text-sm text-slate-400 space-y-1">
              <p>• <span className="text-white">action</span>: "buy", "sell", "long", "short", or "close"</p>
              <p>• <span className="text-white">symbol</span>: "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"</p>
              <p>• <span className="text-white">price</span>: Use TradingView variables like {`{{close}}`} or specific price</p>
              <p>• <span className="text-white">quantity</span>: Amount to trade (optional, defaults to 0.001)</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
