# Trading Bot Setup Guide

## Complete Deliverables

### 1. Deployed Application URLs

**Dashboard (Frontend)**: https://trading-auto-app-byis2mp8.devinapps.com

**Backend API**: https://app-zebaeret.fly.dev

**Webhook URL**: https://app-zebaeret.fly.dev/webhook

**API Documentation**: https://app-zebaeret.fly.dev/docs

### 2. Source Code Location

All source code is located at: `/home/ubuntu/trading-bot-app/`

The repository includes:
- `trading-bot-backend/` - FastAPI backend
- `trading-bot-frontend/` - React frontend
- `README.md` - Comprehensive documentation
- `.gitignore` - Git ignore file

To create a GitHub repository, run:
```bash
cd /home/ubuntu/trading-bot-app
gh auth login
gh repo create YOUR_USERNAME/trading-bot-app --public --source=. --remote=origin --push
```

Or manually create a repository on GitHub and push:
```bash
cd /home/ubuntu/trading-bot-app
git remote add origin https://github.com/YOUR_USERNAME/trading-bot-app.git
git push -u origin devin/1763218790-trading-bot-app
```

## Quick Start (5 Steps)

### Step 1: Access the Dashboard
Go to: https://trading-auto-app-byis2mp8.devinapps.com

### Step 2: Configure Your Exchange
1. Select your exchange (Binance, KuCoin, Kraken, Bybit, or Coinbase)
2. Enter your API Key
3. Enter your API Secret
4. Click "Save Settings"

### Step 3: Copy the Webhook URL
1. Find the webhook URL in the dashboard: `https://app-zebaeret.fly.dev/webhook`
2. Click the copy button

### Step 4: Configure TradingView Alert
1. Go to TradingView
2. Create a new alert
3. Paste the webhook URL: `https://app-zebaeret.fly.dev/webhook`
4. In the Message field, use this JSON:

```json
{
  "action": "buy",
  "symbol": "BTCUSDT",
  "price": "{{close}}",
  "quantity": 0.001
}
```

**Parameters:**
- `action`: "buy", "sell", "long", "short", or "close"
- `symbol`: "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"
- `price`: Use TradingView variables like `{{close}}` or specify a price
- `quantity`: Amount to trade (optional, defaults to 0.001)

### Step 5: Enable Auto-Trading
Toggle "Auto-Trading" to ON in the dashboard

## Testing

### Test the Webhook
Click the "Test Webhook" button in the dashboard to verify the connection.

### Test with TradingView
1. Create a simple alert in TradingView
2. Use the JSON format above
3. Trigger the alert manually
4. Check the dashboard to see the webhook received and order executed

## Dashboard Features

The dashboard displays:
- **API Connection Status**: Connected/Not Connected
- **Auto-Trading Status**: Active/Inactive
- **Total Orders**: Number of orders executed
- **Total PnL**: Profit and loss summary
- **Last Webhook Received**: Timestamp and payload
- **Last Order Executed**: Order details or error messages

## Supported Exchanges

- **Binance**: binance.com
- **KuCoin**: kucoin.com
- **Kraken**: kraken.com
- **Bybit**: bybit.com
- **Coinbase Advanced**: coinbase.com

## Supported Trading Pairs

- BTC/BTCUSDT
- ETH/ETHUSDT
- SOL/SOLUSDT
- XRP/XRPUSDT

## Security Notes

⚠️ **Important Security Information:**

1. **API Keys**: Encrypted with AES-256 before storage
2. **In-Memory Storage**: Keys are stored in-memory and lost on restart
3. **API Permissions**: Only grant trading permissions (no withdrawals)
4. **IP Whitelisting**: Enable on your exchange if supported
5. **Read-Only Keys**: Use read-only keys for testing when possible

## Important Warnings

⚠️ **Risk Warnings:**

1. **Real Money**: This bot executes real trades with real money
2. **Test First**: Always test with small amounts first
3. **Monitor Regularly**: Check the dashboard regularly
4. **Network Issues**: If webhook fails, trades won't execute
5. **Rate Limits**: Be aware of exchange rate limits
6. **Backend Restarts**: If backend restarts, re-enter credentials

## Troubleshooting

### API Connection Failed
- Verify API key and secret are correct
- Check API key has trading permissions
- Ensure IP is whitelisted (if required)
- Try regenerating API keys

### Webhook Not Received
- Verify webhook URL is correct in TradingView
- Check auto-trading is enabled
- Test webhook using "Test Webhook" button
- Check TradingView alert history

### Orders Not Executing
- Ensure auto-trading is enabled
- Verify sufficient balance on exchange
- Check symbol format matches exchange
- Review "Last Order Executed" for errors

### Backend Restarted
If backend restarts:
1. Re-enter API credentials
2. Re-enable auto-trading
3. Webhook URL remains the same

## Redeployment Instructions

### Redeploy Backend
```bash
cd /home/ubuntu/trading-bot-app/trading-bot-backend
# Make changes to app/main.py
# Deploy using Devin deployment tool
```

### Redeploy Frontend
```bash
cd /home/ubuntu/trading-bot-app/trading-bot-frontend
# Make changes to src/App.tsx
npm run build
# Deploy using Devin deployment tool
```

## Extending Features

To add new features:

1. **Add New Exchange**: Update `get_exchange_instance()` in backend
2. **Add New Symbol**: Update `normalize_symbol()` in backend
3. **Add Stop-Loss/Take-Profit**: Implement in webhook handler
4. **Add Persistent Storage**: Replace in-memory state with database
5. **Add User Authentication**: Implement auth middleware

## API Endpoints

Full API documentation: https://app-zebaeret.fly.dev/docs

**Main Endpoints:**
- `POST /set-api-key` - Save exchange credentials
- `POST /webhook` - Receive TradingView alerts
- `POST /toggle-trading` - Enable/disable auto-trading
- `POST /test-webhook` - Test webhook
- `GET /status` - Get current status
- `GET /webhook-url` - Get webhook URL
- `POST /place-order` - Manually place orders
- `POST /close-order` - Close positions

## Technology Stack

**Backend:**
- FastAPI (Python 3.12)
- CCXT (Exchange integration)
- Cryptography (AES-256 encryption)
- Deployed on Fly.io

**Frontend:**
- React 18 + TypeScript
- Vite (Build tool)
- Tailwind CSS (Styling)
- shadcn/ui (UI components)
- Lucide React (Icons)
- Deployed on Devin Apps Platform

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Check the dashboard for error messages
4. Review the comprehensive README.md

## Disclaimer

This software is provided "as is" without warranty. Trading cryptocurrencies carries significant risk. Only trade with money you can afford to lose. The developers are not responsible for any financial losses.
