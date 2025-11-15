# Trading Bot Dashboard - Automated TradingView Alert Execution

A complete plug-and-play automated trading application that executes trades based on TradingView alerts. Simply paste your exchange API keys and webhook URL into TradingView, and the bot handles everything automatically.

## Live Demo

- **Dashboard URL**: https://trading-auto-app-byis2mp8.devinapps.com
- **Backend API**: https://app-zebaeret.fly.dev
- **Webhook URL**: https://app-zebaeret.fly.dev/webhook

## Features

- **Multi-Exchange Support**: Binance, KuCoin, Kraken, Bybit, Coinbase Advanced
- **Secure API Key Storage**: AES-256 encryption for API credentials
- **Auto-Generated Webhook URL**: Automatically generated based on deployment domain
- **Real-Time Dashboard**: Live status updates every 3 seconds
- **Order Execution**: Market/limit orders with stop-loss and take-profit support
- **PnL Tracking**: Track total orders and profit/loss
- **TradingView Integration**: Simple JSON webhook format
- **Auto-Trading Toggle**: Enable/disable automated trading with one click

## Supported Symbols

- BTC/BTCUSDT
- ETH/ETHUSDT
- SOL/SOLUSDT
- XRP/XRPUSDT

## Quick Start Guide

### 1. Access the Dashboard

Navigate to: https://trading-auto-app-byis2mp8.devinapps.com

### 2. Configure Your Exchange API

1. Select your exchange from the dropdown (Binance, KuCoin, Kraken, Bybit, or Coinbase)
2. Enter your API Key
3. Enter your API Secret
4. Click "Save Settings"

The app will verify your credentials and connect to your exchange.

### 3. Copy the Webhook URL

1. Find the "Webhook URL" field in the dashboard
2. Click the copy button next to the URL
3. The webhook URL will be: `https://app-zebaeret.fly.dev/webhook`

### 4. Configure TradingView Alert

1. Go to TradingView and create a new alert
2. In the alert settings, find the "Webhook URL" field
3. Paste the webhook URL: `https://app-zebaeret.fly.dev/webhook`
4. In the "Message" field, use this JSON format:

```json
{
  "action": "buy",
  "symbol": "BTCUSDT",
  "price": "{{close}}",
  "quantity": 0.001
}
```

**Alert JSON Parameters:**
- `action`: "buy", "sell", "long", "short", or "close"
- `symbol`: "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"
- `price`: Use TradingView variables like `{{close}}` or specify a price
- `quantity`: Amount to trade (optional, defaults to 0.001)

### 5. Enable Auto-Trading

1. In the dashboard, toggle "Auto-Trading" to ON
2. The bot will now automatically execute trades when TradingView sends alerts

### 6. Monitor Your Trades

The dashboard shows:
- API Connection Status
- Auto-Trading Status
- Total Orders Executed
- Total PnL
- Last Webhook Received
- Last Order Executed

## Testing

### Test the Webhook

Click the "Test Webhook" button in the dashboard to send a test webhook and verify the connection is working.

### Test with TradingView

1. Create a simple alert in TradingView
2. Use the JSON format above
3. Trigger the alert manually
4. Check the dashboard to see the webhook received and order executed

## Architecture

### Backend (FastAPI)

- **Framework**: FastAPI with Python 3.12
- **Exchange Integration**: CCXT library for unified exchange API
- **Security**: AES-256 encryption for API credentials
- **Deployment**: Fly.io
- **Endpoints**:
  - `POST /set-api-key` - Save and verify exchange credentials
  - `POST /webhook` - Receive TradingView alerts
  - `POST /toggle-trading` - Enable/disable auto-trading
  - `POST /test-webhook` - Test webhook functionality
  - `GET /status` - Get current status
  - `GET /webhook-url` - Get webhook URL
  - `POST /place-order` - Manually place orders
  - `POST /close-order` - Close positions

### Frontend (React + Vite)

- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui
- **Icons**: Lucide React
- **Deployment**: Devin Apps Platform
- **Features**:
  - Real-time status updates (3-second polling)
  - Responsive design
  - Dark theme UI
  - Copy-to-clipboard functionality

## Local Development

### Prerequisites

- Python 3.12+
- Node.js 18+
- Poetry (Python package manager)
- npm (Node package manager)

### Backend Setup

```bash
cd trading-bot-backend

# Install dependencies
poetry install

# Start development server
poetry run fastapi dev app/main.py
```

Backend will run at: http://localhost:8000

### Frontend Setup

```bash
cd trading-bot-frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run at: http://localhost:5173

### Environment Variables

**Backend**: No environment variables needed (uses in-memory storage)

**Frontend** (`.env`):
```
VITE_API_URL=http://localhost:8000
```

For production, update to:
```
VITE_API_URL=https://app-zebaeret.fly.dev
```

## Deployment

### Backend Deployment (Fly.io)

The backend is deployed using the Devin deployment tool:

```bash
# From the project root
deploy backend --dir=/path/to/trading-bot-backend
```

### Frontend Deployment

The frontend is deployed using the Devin deployment tool:

```bash
# Build the frontend first
cd trading-bot-frontend
npm run build

# Deploy
deploy frontend --dir=/path/to/trading-bot-frontend/dist
```

## Security Notes

- API keys are encrypted using AES-256 before storage
- Keys are stored in-memory only (lost on restart)
- For production use, consider using a persistent encrypted database
- Never commit API keys or secrets to the repository
- Use read-only API keys when possible
- Enable IP whitelisting on your exchange if supported

## Important Warnings

⚠️ **Use at Your Own Risk**: This bot executes real trades with real money. Always test with small amounts first.

⚠️ **API Permissions**: Only grant necessary permissions to your API keys (trading only, no withdrawals).

⚠️ **In-Memory Storage**: API keys and state are stored in-memory and will be lost when the backend restarts. You'll need to re-enter your credentials after a restart.

⚠️ **Rate Limits**: The bot respects exchange rate limits, but be aware of your exchange's trading limits.

⚠️ **Network Issues**: If the webhook fails to reach the backend, trades will not execute. Monitor your dashboard regularly.

## Troubleshooting

### API Connection Failed

- Verify your API key and secret are correct
- Check that your API key has trading permissions enabled
- Ensure your IP is whitelisted on the exchange (if required)
- Try regenerating your API keys

### Webhook Not Received

- Verify the webhook URL is correct in TradingView
- Check that auto-trading is enabled in the dashboard
- Test the webhook using the "Test Webhook" button
- Check TradingView alert history to see if alerts are firing

### Orders Not Executing

- Ensure auto-trading is enabled
- Verify you have sufficient balance on the exchange
- Check the symbol format matches your exchange
- Review the "Last Order Executed" section for error messages

### Backend Restarted

If the backend restarts, you'll need to:
1. Re-enter your API credentials
2. Re-enable auto-trading
3. The webhook URL remains the same

## API Documentation

Full API documentation is available at: https://app-zebaeret.fly.dev/docs

## Technology Stack

**Backend:**
- FastAPI
- CCXT (Cryptocurrency Exchange Trading Library)
- Cryptography (AES-256 encryption)
- Python 3.12

**Frontend:**
- React 18
- TypeScript
- Vite
- Tailwind CSS
- shadcn/ui
- Lucide React

**Deployment:**
- Backend: Fly.io
- Frontend: Devin Apps Platform

## Contributing

This is a proof-of-concept application. For production use, consider:
- Adding persistent database storage
- Implementing user authentication
- Adding more sophisticated order management
- Implementing proper error handling and retry logic
- Adding comprehensive logging
- Setting up monitoring and alerts

## License

MIT License - Use at your own risk

## Support

For issues or questions, please create an issue in the GitHub repository.

## Disclaimer

This software is provided "as is" without warranty of any kind. Trading cryptocurrencies carries significant risk. Only trade with money you can afford to lose. The developers are not responsible for any financial losses incurred while using this software.
